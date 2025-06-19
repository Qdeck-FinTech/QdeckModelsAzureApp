import azure.functions as func
import azure.durable_functions as df
import logging

import os
import sys
import inspect
import time
import json

from pythonnet import load, set_runtime
# Tell pythonnet to use Mono
set_runtime("coreclr")

import clr

load("coreclr")

# add the path to the Mercury bin directory
root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
bin_dir = os.path.join(root_dir, "mercury")
sys.path.insert(0, bin_dir)

clr.AddReference("Mercury")

from Mercury import MercuryRunner, PythonLoggerAdapter

from logger.net_logger import net_logger
from configuration.configuration import runner_config

from AlphaVee.MercurySystem import AVModelRunner
from MasterSystemLive.MercurySystem import MLModelRunner
from NasdaqDorseyWright.MercurySystem import NDWModelRunner
from PassiveIndex.MercurySystem import PassiveIndexModelRunner
from KeebeckMultiStrategy.MercurySystem import KeebeckMultiStrategyModelRunner
from DirectIndexing.MercurySystem import DirectIndexingModelRunner


class QdeckModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

    # def __init__(self):
    #     super().__init__()

    # def __init__(self, logger=None, configJsonString=None):
    #     logging.info("Initializing QdeckModelRunner...", configJsonString)
    #     super().__init__(logger, configJsonString)

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
                    print("Unknown runner" + mod)

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
            logging.info("Running model: ", str(model_id) + " ...")
            runId = model_runner.run_model(model_id, update_qdeck, live, config)

            logging.info("Model ", str(model_id), " run completed!")
        else:
            logging.info("No model ", str(model_id), " run. No model details loaded.")

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
    payload = input_data.get("data")

    logging.info(f"qdeck_model_orchestrator(): {function_name} {payload}")

    model_id = payload.get("model_id", 0)

    if model_id <= 0:
        raise Exception("Invalid model_id provided in input.")

    live = payload.get("live", False)
    config = payload.get("config", False)

    tasks = []
    results = []

    if function_name == "run_model":
        task = context.call_activity(
            "run_model", {"model_id": model_id, "live": live, "config": config}
        )
        tasks.append(task)

        # wait for all tasks to complete
        results = yield context.task_all(tasks)

    elif function_name == "run_all_models":
        
        logging.info(f"qdeck_model_orchestrator() run_all_models mercury config: {runner_config}")        
        
        # init model runner
        runner = QdeckModelRunner(net_logger, runner_config)

        # Ggt the scheduled model IDs
        model_ids = runner.get_scheduled_model_ids()

        if len(model_ids) > 0:
            # init run_all notification service
            runner.init_run_all(model_ids)

            for model_id in model_ids:
                task = context.call_activity(
                    "run_model", {"model_id": model_id, "live": True, "config": config}
                )
                tasks.append(task)

            # wait for all tasks to complete
            results = yield context.task_all(tasks)

            # update status for each model
            for result in results:
                model_id = result.get("model_id", 0)
                run_id = result.get("run_id", 0)

                logging.info(f"Updating model run complete: {model_id}, {run_id}")
                # Update the model run status in the notification service
                runner.update_model_run_complete(model_id, run_id)

            # complete run_all notification service
            runner.complete_run_all()

    return results


# Activity functions
@myApp.activity_trigger(input_name="context")
def run_model(context):
    model_id = context.get("model_id")
    live = context.get("live", False)
    config = context.get("config", "")

    logging.info(f"run_model(): {model_id} {live} {config}")

    logging.info(f"run_model() mercury config: {runner_config}")

    run_id = 0

    if model_id > 0:
        modelRunner = QdeckModelRunner(net_logger, runner_config)
        run_id = modelRunner.run_model(model_id, False, live, config)

    return {"model_id": model_id, "run_id": run_id}


