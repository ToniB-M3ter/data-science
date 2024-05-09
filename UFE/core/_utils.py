import uuid
from datetime import datetime


def datetime_to_m3ter_format(date_time: datetime) -> str:
    """Converts datetime object to m3ter accepted format (ISO8601)

    Args:
        date_time (datetime): Datetime object created using built in datetime library

    Returns:
        str: Datetime in ISO8601 format
    """
    return date_time.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_uid() -> str:
    """Genereate a uuid string acceptad by m3ter

    Returns:
        str: 4 partition 36 char length uuid
    """
    return str(uuid.uuid4())
