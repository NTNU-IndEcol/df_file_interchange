"""
Base class for structured custom info

Make sure any structured custom info is derived from this.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
)


class FIBaseCustomInfo(BaseModel):
    """Wrapper class to store user custom info

    N.B. This, and any descendent, MUST be able to deserialise based on a
    provided dictionary!

    A descendent of this is usually supplied as an object when writing a file to
    include additional metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unstructured_data: dict = {}

    @classmethod
    def get_classname(cls) -> str:
        """Returns the classname

        Fairly prosaic but we use this for serialization and deserialization.

        Returns
        -------
        str
            The classname.
        """
        return cls.__name__

    @computed_field
    @property
    def classname(self) -> str:
        """Ensures classname is included in serialization

        By defining this as a computed Pydantic attribute, it's included in any
        serialization.

        Returns
        -------
        str
            Our classname
        """

        return self.get_classname()
