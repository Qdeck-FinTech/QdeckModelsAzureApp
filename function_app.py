import azure.functions as func
import azure.durable_functions as df
import logging

import os
import sys
import inspect
import time
import json


from pythonnet import load, set_runtime

set_runtime("coreclr")
load("coreclr")

import clr

# add the path to the Mercury bin directory
root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
bin_dir = os.path.join(root_dir, "mercury")
sys.path.insert(0, bin_dir)

clr.AddReference("Mercury")

from Mercury import MercuryRunner

from System.Collections.Generic import List
from System import Int32

from logger.net_logger import net_logger
from configuration.configuration import QdeckModelRunnerConfiguration

from AlphaVee.MercurySystem import AVModelRunner
from MasterSystemLive.MercurySystem import MLModelRunner
from NasdaqDorseyWright.MercurySystem import NDWModelRunner
from PassiveIndex.MercurySystem import PassiveIndexModelRunner
from KeebeckMultiStrategy.MercurySystem import KeebeckMultiStrategyModelRunner
from DirectIndexing.MercurySystem import DirectIndexingModelRunner


class QdeckModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

    def get_model_details(self, model_id):
        model_run_details = None
        if model_id > 0:
            # load configuration from database
            cfg_data_json = self.load_model_run_details(str(model_id))

            if cfg_data_json is not None:
                model_run_details = json.loads(cfg_data_json)

        return model_run_details

    def get_system_runner(self, model_run_details=None):
        runner = None

        runner_config = QdeckModelRunnerConfiguration().get_net_config()

        if model_run_details is not None:
            mod = model_run_details["folder"]
            match mod:
                case "AlphaVee":
                    runner = AVModelRunner(net_logger, runner_config)
                case "MasterSystemLive":
                    runner = MLModelRunner(net_logger, runner_config)
                case "NasdaqDorseyWright":
                    runner = NDWModelRunner(net_logger, runner_config)
                case "PassiveIndex":
                    runner = PassiveIndexModelRunner(net_logger, runner_config)
                case "KeebeckMultiStrategy":
                    runner = KeebeckMultiStrategyModelRunner(net_logger, runner_config)
                case "DirectIndexing":
                    runner = DirectIndexingModelRunner(net_logger, runner_config)
                case _:
                    logging.info("Unknown runner" + str(mod))

        return runner

    def run_model(self, model_id=0, update_qdeck=0, live=0, config=None):
        runId = 0
        model_run_details = None
        if model_id > 0:
            model_run_details = self.get_model_details(model_id)

        # get system runner
        model_runner = self.get_system_runner(model_run_details)

        if model_runner is not None:
            # run model
            logging.info("Running model: " + str(model_id) + " ...")
            runId = model_runner.run_model(model_id, update_qdeck, live, config)

            logging.info("Model " + str(model_id) + " run completed!")
        else:
            logging.info("No model " + str(model_id) + " run. No model details loaded.")

        model_runner = None

        return runId


myApp = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)


# An HTTP-triggered function with a Durable Functions client binding
@myApp.route(
    route="orchestrators/{functionName}",
    methods=[func.HttpMethod.POST],
    auth_level=func.AuthLevel.FUNCTION,
)
@myApp.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    function_name = req.route_params.get("functionName")

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    # Combine into a single object for the orchestrator
    payload = {"functionName": function_name, "data": body}

    instance_id = await client.start_new("qdeck_model_orchestrator", None, payload)

    logging.info(f"http_start(): {function_name} {instance_id}")

    response = client.create_check_status_response(req, instance_id)
    return response


# Orchestrator
@myApp.orchestration_trigger(context_name="context")
def qdeck_model_orchestrator(context):
    input_data = context.get_input()

    function_name = input_data.get("functionName")
    payload = input_data.get("data", {})

    logging.info(f"qdeck_model_orchestrator(): {function_name} {payload}")

    tasks = []
    results = []

    if function_name == "run_model":
        model_id = payload.get("model_id")

        # If model_id is missing or not positive, skip execution
        if not isinstance(model_id, int) or model_id <= 0:
            logging.warning(f"Invalid or missing model_id: {model_id}")
            return []

        live = payload.get("live", False)
        config = payload.get("config", None)

        # Fan-out task
        task = context.call_activity(
            "run_model", {"model_id": model_id, "live": live, "config": config}
        )

        # wait for tasks to complete
        results = yield context.task_all([task])

    elif function_name == "run_all_models":
        logging.info("qdeck_model_orchestrator() run_all_models")

        # Step 1: Call an activity to fetch model IDs
        model_ids = yield context.call_activity("get_scheduled_model_ids")

        if len(model_ids) > 0:
            # Step 2: Fan-out model runs
            tasks = [
                context.call_activity(
                    "run_model", {"model_id": mid, "live": True, "config": None}
                )
                for mid in model_ids
            ]
            results = yield context.task_all(tasks)

            # Step 3: Complete notification (activity)
            yield context.call_activity(
                "complete_run_all", {"model_ids": model_ids, "results": results}
            )

    return results


# Activity functions
@myApp.activity_trigger(input_name="context")
def run_model(context):
    model_id = context.get("model_id")
    live = context.get("live", False)
    config = context.get("config", "")

    logging.info(f"run_model(): {model_id} {live} {config}...")

    runner_config = QdeckModelRunnerConfiguration().get_net_config()

    run_id = 0

    if model_id > 0:
        modelRunner = QdeckModelRunner(net_logger, runner_config)
        run_id = modelRunner.run_model(model_id, False, live, config)

    return {"model_id": model_id, "run_id": run_id}


@myApp.activity_trigger(input_name="input")
def get_scheduled_model_ids(input):
    logging.info("get_scheduled_model_ids()...")

    runner_config = QdeckModelRunnerConfiguration().get_net_config()

    runner = QdeckModelRunner(net_logger, runner_config)
    ids = runner.get_scheduled_model_ids()

    return list(ids)  # Ensure we return a list of model IDs


@myApp.activity_trigger(input_name="context")
def complete_run_all(context):
    model_ids = context.get("model_ids")
    results = context.get("results")

    logging.info("complete_run_all() ...")

    # get runner config
    runner_config = QdeckModelRunnerConfiguration().get_net_config()

    # create a new instance of the QdeckModelRunner
    runner = QdeckModelRunner(net_logger, runner_config)

    # initialize the model list for the email
    runner.get_scheduled_model_ids()

    # initialize the list of model IDs for .NET
    dotnet_model_ids = List[Int32]()
    for item in model_ids:
        dotnet_model_ids.Add(Int32(item))

    # initialize the run all process
    runner.init_run_all(dotnet_model_ids)

    # update status for each model run
    for result in results:
        model_id = result.get("model_id", 0)
        run_id = result.get("run_id", 0)

        if model_id > 0 and run_id > 0:
            runner.update_model_run_complete(model_id, run_id)

    # complete the run all process
    runner.complete_run_all()

    return {
        "status": "Completed",
        "message": "Email notification sent.",
        "model_ids": model_ids,
    }
