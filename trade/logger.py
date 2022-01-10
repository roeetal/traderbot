from logging import Handler, getLevelName

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a slack channel.
    """

    def __init__(self, token: str, channel: str, username: str, fmt=None):
        """
        Initialize the handler.
        """
        Handler.__init__(self)
        self.channel = channel
        self.username = username
        self.client = WebClient(token=token)
        if fmt:
            self.setFormatter(fmt)

    def _write(self, message: str):
        try:
            self.client.chat_postMessage(channel=self.channel, text=message)
        except SlackApiError as e:
            print(f"Slack API error: {e.response['error']}")

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the Slack channel.
        """
        try:
            msg = self.format(record)
            self._write(msg)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s %s-%s(%s)>' % (self.__class__.__name__, self.channel, self.username, level)
