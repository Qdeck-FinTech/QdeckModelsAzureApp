import json
import configargparse
from dotenv import dotenv_values

import clr
from System import String  # type: ignore
from System.Collections.Generic import Dictionary  # type: ignore


def dot_notation_to_nested_dict(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Convert booleans and ints
        if isinstance(value, str):
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
        d[parts[-1]] = value
    return nested_dict


def flatten_dict(d, parent_key='', sep=':'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)
    return items

def create_dotnet_config_dic(config_dict):
    # Convert Python dict to .NET Dictionary<string, string>
    net_dict = Dictionary[String, String]()
    for k, v in config_dict.items():
        net_dict[k] = v

    return net_dict


class QdeckModelRunnerConfiguration:
    def __init__(self, env_path: str = ".env"):
        # Step 1: Load and convert .env into nested dict
        self.flat_env = dotenv_values(env_path)
        self.nested_env = dot_notation_to_nested_dict(self.flat_env)

        # Step 2: Create parser and bind all .env keys
        self.parser = configargparse.ArgParser()

        # Use is_config_file for optional external config file support
        self.parser.add_argument(
            "-s",
            "--settings_path",
            help="settings file",
            default=None,
            is_config_file=True,
            env_var="SETTINGS_PATH",
        )

        # Register arguments for all known .env fields
        self.parser.add_argument(
            "--SQLSettings.refinitivDataSQL",
            env_var="SQLSettings.refinitivDataSQL",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings.qdeckSqlConnectionString",
            env_var="SQLSettings.qdeckSqlConnectionString",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings.qdeckPostgresConnectionString",
            env_var="SQLSettings.qdeckPostgresConnectionString",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings.usePostgres", env_var="SQLSettings.usePostgres", type=str
        )

        self.parser.add_argument(
            "--StatusNotificationSettings.topic",
            env_var="StatusNotificationSettings.topic",
            type=str,
        )
        self.parser.add_argument(
            "--StatusNotificationSettings.address",
            env_var="StatusNotificationSettings.address",
            type=str,
        )
        self.parser.add_argument(
            "--StatusNotificationSettings.port",
            env_var="StatusNotificationSettings.port",
            type=int,
        )

        self.parser.add_argument(
            "--LiveRunSettings.externalModelBaseModelID",
            env_var="LiveRunSettings.externalModelBaseModelID",
            type=int,
        )
        self.parser.add_argument(
            "--LiveRunSettings.liveSocketPortStart",
            env_var="LiveRunSettings.liveSocketPortStart",
            type=int,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckLoginAddress",
            env_var="LiveRunSettings.qdeckLoginAddress",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckLoginUser",
            env_var="LiveRunSettings.qdeckLoginUser",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckLoginPass",
            env_var="LiveRunSettings.qdeckLoginPass",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckLoginGuid",
            env_var="LiveRunSettings.qdeckLoginGuid",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckResultsAddress",
            env_var="LiveRunSettings.qdeckResultsAddress",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckProductionFolderRoot",
            env_var="LiveRunSettings.qdeckProductionFolderRoot",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckOutputFolderRoot",
            env_var="LiveRunSettings.qdeckOutputFolderRoot",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings.qdeckOutputFolderFormat",
            env_var="LiveRunSettings.qdeckOutputFolderFormat",
            type=str,
        )

        self.parser.add_argument(
            "--ScheduledRunSettings.sendEmail",
            env_var="ScheduledRunSettings.sendEmail",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.server",
            env_var="ScheduledRunSettings.server",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.port", env_var="ScheduledRunSettings.port", type=int
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.from", env_var="ScheduledRunSettings.from", type=str
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.to", env_var="ScheduledRunSettings.to", type=str
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.username",
            env_var="ScheduledRunSettings.username",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings.password",
            env_var="ScheduledRunSettings.password",
            type=str,
        )

    def get_args(self):
        return self.parser.parse_args()

    def get_nested_config(self):
        return self.nested_env

    def get_config(self):
        config, _ = self.parser.parse_known_args()
        return config

    def get_net_config(self):
        return json.dumps(self.nested_env)



env_config = QdeckModelRunnerConfiguration()
runner_config = env_config.get_net_config()