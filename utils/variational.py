import numpy as np
import os, sys, inspect

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_lattice as tfl

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers

from tqdm import tqdm


def get_logistic_marginal_pdf(data, scale):
    log_scale = -tf.math.log(scale)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1, data.shape[-1]], dtype=tf.float32)
        ]
    )
    def logistic_pdf(x):
        delta = (x - data) / scale
        return tf.reduce_mean(
            tf.math.exp(log_scale + delta - 2 * tf.nn.softplus(delta)), axis=1
        )

    return logistic_pdf


def get_logistic_marginal_cdf(data, scale):
    # make a tensor product for speed?
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1, data.shape[-1]], dtype=tf.float32)
        ]
    )
    def logistic_cdf(x):
        delta = (x - data) / scale
        return tf.reduce_mean(tf.nn.sigmoid(delta), axis=1)

    return logistic_cdf


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)],
    jit_compile=True,
)
def quantile_transform(data):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)],
        jit_compile=True,
    )
    def vec_quantile(x):
        return tf.reduce_mean(tf.sign(tf.maximum(x - data, 0.0)), axis=0)

    return tf.map_fn(fn=vec_quantile, elems=data, parallel_iterations=24)


def get_pwl_cdf_pair(
    data,
    n_keypoints=300,
    batch_size=1024,
    epochs=200,
    learning_rate=0.00001,
    pareto_fit_fraction=0.05,
    pareto_tail_parameter=0.001,
    verbose=0,
):
    # build a dataset object for training
    data = data.copy()
    quantiles = quantile_transform(data)
    event_shape = data.shape[-1]

    cdf_train = (
        tf.data.Dataset.from_tensor_slices((data, quantiles))
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # use pareto statistics to estimate appropriate bounds for each model
    pareto_ratio = pareto_tail_parameter / pareto_fit_fraction

    right_tail = tf.math.top_k(data.T, int(len(data) * pareto_fit_fraction)).values
    beta_max = tf.reduce_min(right_tail, axis=1, keepdims=True)
    max_val = beta_max[:, 0] * (pareto_ratio) ** tf.reduce_mean(
        tf.math.log(beta_max / right_tail), axis=1
    )

    left_tail = tf.math.top_k(-data.T, int(len(data) * pareto_fit_fraction)).values
    beta_min = tf.reduce_min(left_tail, axis=1, keepdims=True)
    min_val = -beta_min[:, 0] * (pareto_ratio) ** tf.reduce_mean(
        tf.math.log(beta_min / left_tail), axis=1
    )

    # create and fit the CDF model for each marginal
    pwl_cdf = tfl.layers.ParallelCombination(
        calibration_layers=[
            tfl.layers.PWLCalibration(
                units=1,
                input_keypoints=sorted(
                    np.concatenate(
                        [
                            np.random.choice(data[:, i], n_keypoints - 2, replace=False)
                            + np.random.normal(0, 0.0001, n_keypoints - 2),
                            [min_val[i], max_val[i]],
                        ],
                        axis=0,
                    )
                ),
                output_min=0.0,
                output_max=1.0,
                clamp_min=True,
                clamp_max=True,
                dtype=tf.float32,
                monotonicity="increasing",
                input_keypoints_type="learned_interior",
            )
            for i in range(event_shape)
        ]
    )

    cdf_model = tfk.Sequential(
        [
            tfkl.Input(event_shape, dtype=tf.float32),
            pwl_cdf,
        ]
    )

    cdf_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tfk.losses.MeanSquaredError(),
        steps_per_execution=int(len(data) / batch_size) + 1,
    )
    cdf_model.fit(cdf_train, epochs=epochs, verbose=verbose)

    # use the tail values and CDF model to directly construct the inverse CDF model
    pwl_icdf = tfl.layers.ParallelCombination(
        calibration_layers=[
            tfl.layers.PWLCalibration(
                units=1,
                input_keypoints=np.linspace(
                    0.0, 1.0, num=n_keypoints, dtype=np.float32
                ),
                output_min=min_val[i],
                output_max=max_val[i],
                dtype=tf.float32,
                monotonicity="increasing",
                input_keypoints_type="learned_interior",
            )
            for i in range(event_shape)
        ]
    )

    icdf_model = tfk.Sequential(
        [
            tfkl.Input(event_shape, dtype=tf.float32),
            pwl_icdf,
        ]
    )

    for i, cal_layer in enumerate(pwl_cdf.calibration_layers):
        new_lengths = np.log(cal_layer.get_weights()[1][1:].T + 0.00000001)
        new_heights = np.concatenate(
            [
                np.ones([1, 1]) * cal_layer._keypoint_min,
                (
                    tf.nn.softmax(cal_layer.get_weights()[0], axis=1)
                    * cal_layer._keypoint_range
                )
                .numpy()
                .T,
            ],
            axis=0,
        )
        pwl_icdf.calibration_layers[i].set_weights((new_lengths, new_heights))

    return cdf_model, icdf_model


def get_kl_loss(z_map):
    def kldiv(y_true, y_pred):
        if isinstance(z_map, list):
            losses = [loss for layer in z_map for loss in layer.losses]
        else:
            losses = z_map.losses
        return tf.add_n(losses)

    return kldiv


def prob_loglik(x, rv_x):
    return -tf.reduce_mean(rv_x.log_prob(x), axis=0)


# functions for constructing variational models


def build_encoder(
    input_shape,
    width,
    depth,
    encoded_size,
    activation_fx=tf.nn.sigmoid,
    pre_embedding=None,
):
    x_in = tfkl.Input(shape=input_shape)
    if pre_embedding is not None:
        x = pre_embedding(x_in)
    else:
        x = x_in

    x_res = None
    for _ in range(depth):
        x = tfkl.Dense(width, activation=activation_fx, dtype=tf.float32)(x)
        if x_res is not None:
            x += x_res
        x_res = x

    z_hat = tfkl.Dense(
        tfpl.MultivariateNormalTriL.params_size(encoded_size),
        activation=None,
        dtype=tf.float32,
    )(x)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(encoded_size))
    z_map = tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True),
        dtype=tf.float32,
    )
    z = z_map(z_hat)

    return tfk.Model(inputs=[x_in], outputs=z), z_map, prior


def build_gibbs_encoder(
    input_shape,
    width,
    depth,
    encoded_size,
    activation_fx=tf.nn.sigmoid,
    pre_embedding=None,
):
    x_in = tfkl.Input(shape=input_shape)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(1))

    if pre_embedding is not None:
        x = pre_embedding(x_in)
    else:
        x = x_in

    x_res = None
    for _ in range(depth):
        x = tfkl.Dense(width, activation=activation_fx, dtype=tf.float32)(x)
        if x_res is not None:
            x += x_res
        x_res = x

    z_layers = []
    latent = []
    for _ in range(encoded_size):
        if len(latent) > 0:
            x = tfkl.Dense(width, activation=activation_fx, dtype=tf.float32)(
                tfkl.Concatenate()([x, z])
            )
        z_hat = tfkl.Dense(2, activation=None, dtype=tf.float32)(x)
        z_map = tfpl.MultivariateNormalTriL(
            1,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True),
            dtype=tf.float32,
        )
        z_layers.append(z_map)
        z = z_map(z_hat)
        latent.append(z)

    output_layer = tfkl.Concatenate()(latent)

    return tfk.Model(inputs=[x_in], outputs=output_layer), z_layers, prior


def build_cvae_decoder(
    input_shape,
    target_shape,
    width,
    depth,
    encoded_size,
    sigma,
    activation_fx=tf.nn.sigmoid,
    post_embedding=None,
    discrete=False,
):
    x_in = tfkl.Input(shape=input_shape)
    z_in = tfkl.Input(shape=encoded_size)
    combined_latent = tfkl.Concatenate()([x_in, z_in])

    y = combined_latent
    y_res = None
    for _ in range(depth):
        y = tfkl.Dense(width, activation=activation_fx, dtype=tf.float32)(y)
        if y_res is not None:
            y += y_res
        y_res = y

    if post_embedding is not None:
        y = tfkl.Dense(
            input_shape + target_shape, activation=tf.nn.sigmoid, dtype=tf.float32
        )(y)
        y = post_embedding(y)[..., :-input_shape]
    else:
        y = tfkl.Dense(target_shape, dtype=tf.float32)(y)

    if discrete:
        y_hat = tfpl.DistributionLambda(lambda t: tfd.Bernoulli(logits=t))(y)
    else:
        y_hat = tfpl.DistributionLambda(lambda t: tfd.Normal(t, sigma))(
            y
        )  # is this scaled differently than if we used MVN?

    return tfk.Model(inputs=[x_in, z_in], outputs=y_hat)


def build_vae_decoder(
    input_shape,
    width,
    depth,
    encoded_size,
    sigma,
    activation_fx=tf.nn.sigmoid,
    post_embedding=None,
    discrete=False,
):
    z_in = tfkl.Input(shape=encoded_size)

    y = z_in
    y_res = None
    for _ in range(depth):
        y = tfkl.Dense(width, activation=activation_fx, dtype=tf.float32)(y)
        if y_res is not None:
            y += y_res
        y_res = y

    if post_embedding is not None:
        y = tfkl.Dense(input_shape, activation=tf.nn.sigmoid, dtype=tf.float32)(y)
        y = post_embedding(y)
    else:
        y = tfkl.Dense(input_shape, dtype=tf.float32)(y)

    if discrete:
        y_hat = tfpl.DistributionLambda(lambda t: tfd.Bernoulli(logits=t))(y)
    else:
        y_hat = tfpl.DistributionLambda(lambda t: tfd.Normal(t, sigma))(
            y
        )  # is this scaled differently than if we used MVN?

    return tfk.Model(inputs=z_in, outputs=y_hat)


def build_cvae(
    input_shape,
    target_shape,
    width,
    encoder_depth,
    encoded_size,
    decoder_depth,
    sigma,
    activation_fx=tf.nn.sigmoid,
    pre_embedding=None,
    post_embedding=None,
    discrete=False,
):
    x_in = tfkl.Input(shape=input_shape)
    y_in = tfkl.Input(shape=target_shape)
    combined_in = tfkl.Concatenate()([y_in, x_in])

    encoder, z_map, prior = build_encoder(
        input_shape + target_shape,
        width,
        encoder_depth,
        encoded_size,
        activation_fx=activation_fx,
        pre_embedding=pre_embedding,
    )
    decoder = build_cvae_decoder(
        input_shape,
        target_shape,
        width,
        decoder_depth,
        encoded_size,
        sigma,
        activation_fx=activation_fx,
        post_embedding=post_embedding,
        discrete=discrete,
    )

    return (
        tfk.Model(inputs=[x_in, y_in], outputs=decoder([x_in, encoder(combined_in)])),
        encoder,
        decoder,
        z_map,
        prior,
    )


def build_vae(
    input_shape,
    width,
    encoder_depth,
    encoded_size,
    decoder_depth,
    sigma,
    activation_fx=tf.nn.sigmoid,
    pre_embedding=None,
    post_embedding=None,
    discrete=False,
):
    x_in = tfkl.Input(shape=input_shape)

    encoder, z_map, prior = build_encoder(
        input_shape,
        width,
        encoder_depth,
        encoded_size,
        activation_fx=activation_fx,
        pre_embedding=pre_embedding,
    )
    decoder = build_vae_decoder(
        input_shape,
        width,
        decoder_depth,
        encoded_size,
        sigma,
        activation_fx=activation_fx,
        post_embedding=post_embedding,
        discrete=discrete,
    )

    return (
        tfk.Model(inputs=x_in, outputs=decoder(encoder(x_in))),
        encoder,
        decoder,
        z_map,
        prior,
    )


if __name__ == "__main__":
    from System import DateTime  # type: ignore

    import matplotlib.pyplot as plt
    import seaborn as sns

    # hyperparameters
    buffer_size = 150
    target_symbols = {"TY", "US", "FV", "DX", "CL", "GC"}
    pred_symbols = {"SQ"}
    symbols = target_symbols | pred_symbols
    start = DateTime(2000, 1, 1)
    end = DateTime(2022, 1, 1)
    input_shape = len(pred_symbols)
    target_dims = len(target_symbols)

    adj_prices = get_adjusted_prices(symbols, start, end)
    adj_returns = np.log(1 + adj_prices.pct_change().dropna())
    cvae_df = adj_returns[list(target_symbols) + list(pred_symbols)].astype(np.float32)

    # make CVAE kernel data to train model
    # cvae_kernel_data = np.concatenate([cdf_model(x) for x in tqdm(np.array_split(cvae_df.values,100))],axis=0)
    # cvae_kernel_df = pd.DataFrame(cvae_kernel_data,index=cvae_df.index,columns=cvae_df.columns)

    # g = sns.PairGrid(cvae_kernel_df,diag_sharey=True)
    # g.map_diag(sns.kdeplot,cut=0)
    # g.map_upper(sns.scatterplot,alpha=0.25,s=3,edgecolor="None")
    # # g.map_lower(sns.kdeplot,cut=0)
    # plt.show()

    # make cdf model pair
    cdf_model, icdf_model = get_pwl_cdf_pair(cvae_df.values, n_keypoints=1000)

    # cdf plot
    for cal_layer in cdf_model.layers[0].calibration_layers:
        plt.scatter(
            cal_layer.keypoints_inputs().numpy()[:, 0],
            cal_layer.keypoints_outputs().numpy()[:, 0],
        )
    plt.legend(cvae_df.columns)
    plt.show()

    # quantile plot
    test_samples = np.concatenate(
        [cdf_model(x).numpy() for x in np.array_split(cvae_df.values, 100)], axis=0
    )
    for samples in test_samples.T:
        sns.kdeplot(samples, bw_adjust=1, cut=0)
    #     sns.histplot(samples,bins=200)
    plt.legend(cvae_df.columns)
    plt.show()

    # icdf plot
    for cal_layer in icdf_model.layers[0].calibration_layers:
        plt.scatter(
            cal_layer.keypoints_inputs().numpy()[:, 0],
            cal_layer.keypoints_outputs().numpy()[:, 0],
        )
    plt.legend(cvae_df.columns)
    plt.show()

    # implied icdf marginal plot
    fig, axes = plt.subplots(
        ncols=int(len(cvae_df.columns) ** 0.5 + 1),
        nrows=int(len(cvae_df.columns) ** 0.5 + 1),
        gridspec_kw={"hspace": 0.2},
    )
    axes = axes.flatten()
    test_samples = np.concatenate(
        [icdf_model(tfd.Uniform().sample([30, 7])).numpy() for _ in range(100)], axis=0
    )
    for i, col in enumerate(cvae_df.columns):
        sns.kdeplot(test_samples.T[i], ax=axes[i])
        sns.kdeplot(cvae_df[col], ax=axes[i])
        axes[i].legend(["marginal_fit", "raw"])
    plt.show()
