import base64

class EncryptionUtil:
    """
    Utility class for Base64 encoding and decoding of UTF-8 strings.
    """

    @staticmethod
    def encode(text: str) -> str:
        """
        Encodes a plain text string into a Base64-encoded string.

        Args:
            text (str): The input plain text string.

        Returns:
            str: The Base64-encoded string.
        """
        return base64.b64encode(text.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode(encoded_text: str) -> str:
        """
        Decodes a Base64-encoded string into plain text.

        Args:
            encoded_text (str): The Base64-encoded string.

        Returns:
            str: The decoded plain text string.
        """
        return base64.b64decode(encoded_text).decode('utf-8')
