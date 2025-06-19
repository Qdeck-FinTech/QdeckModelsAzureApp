import json
import configargparse


def dot_notation_to_nested_dict(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split("_")
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


def flatten_dict(d, parent_key="", sep=":"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)
    return items


class QdeckModelRunnerConfiguration:
    def __init__(self):
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
            "--SQLSettings_refinitivDataSQL",
            env_var="SQLSettings_refinitivDataSQL",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings_qdeckSqlConnectionString",
            env_var="SQLSettings_qdeckSqlConnectionString",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings_qdeckPostgresConnectionString",
            env_var="SQLSettings_qdeckPostgresConnectionString",
            type=str,
        )
        self.parser.add_argument(
            "--SQLSettings_usePostgres", env_var="SQLSettings_usePostgres", type=str
        )

        self.parser.add_argument(
            "--StatusNotificationSettings_topic",
            env_var="StatusNotificationSettings_topic",
            type=str,
        )
        self.parser.add_argument(
            "--StatusNotificationSettings_address",
            env_var="StatusNotificationSettings_address",
            type=str,
        )
        self.parser.add_argument(
            "--StatusNotificationSettings_port",
            env_var="StatusNotificationSettings_port",
            type=int,
        )

        self.parser.add_argument(
            "--LiveRunSettings_externalModelBaseModelID",
            env_var="LiveRunSettings_externalModelBaseModelID",
            type=int,
        )
        self.parser.add_argument(
            "--LiveRunSettings_liveSocketPortStart",
            env_var="LiveRunSettings_liveSocketPortStart",
            type=int,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckLoginAddress",
            env_var="LiveRunSettings_qdeckLoginAddress",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckLoginUser",
            env_var="LiveRunSettings_qdeckLoginUser",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckLoginPass",
            env_var="LiveRunSettings_qdeckLoginPass",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckLoginGuid",
            env_var="LiveRunSettings_qdeckLoginGuid",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckResultsAddress",
            env_var="LiveRunSettings_qdeckResultsAddress",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckProductionFolderRoot",
            env_var="LiveRunSettings_qdeckProductionFolderRoot",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckOutputFolderRoot",
            env_var="LiveRunSettings_qdeckOutputFolderRoot",
            type=str,
        )
        self.parser.add_argument(
            "--LiveRunSettings_qdeckOutputFolderFormat",
            env_var="LiveRunSettings_qdeckOutputFolderFormat",
            type=str,
        )

        self.parser.add_argument(
            "--ScheduledRunSettings_sendEmail",
            env_var="ScheduledRunSettings_sendEmail",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_server",
            env_var="ScheduledRunSettings_server",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_port", env_var="ScheduledRunSettings_port", type=int
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_from", env_var="ScheduledRunSettings_from", type=str
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_to", env_var="ScheduledRunSettings_to", type=str
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_username",
            env_var="ScheduledRunSettings_username",
            type=str,
        )
        self.parser.add_argument(
            "--ScheduledRunSettings_password",
            env_var="ScheduledRunSettings_password",
            type=str,
        )

    def get_args(self):
        return self.parser.parse_args()

    def get_config(self):
        config, _ = self.parser.parse_known_args()
        return config

    def get_nested_config(self):
        config = self.get_config()
        flat_config = {k: v for k, v in vars(config).items() if v is not None}

        nested_env = dot_notation_to_nested_dict(flat_config)
        return nested_env

    def get_net_config(self):
        nested_env = self.get_nested_config()
        return json.dumps(nested_env)
