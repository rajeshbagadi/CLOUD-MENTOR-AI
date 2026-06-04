class CloudMentorError(Exception):
    """Base exception class for CloudMentor AI."""
    pass


class DocumentLoadError(CloudMentorError):
    """Exception raised when document loading fails."""
    pass


class UnsupportedFileTypeError(DocumentLoadError):
    """Exception raised when file type is not supported."""
    pass


class EmptyDocumentError(DocumentLoadError):
    """Exception raised when a document contains no text content."""
    pass
