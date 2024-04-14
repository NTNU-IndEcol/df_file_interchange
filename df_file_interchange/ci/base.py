"""
'Base' classes for column spec and optional info.

"""

from pydantic import (
    BaseModel,
)


class FIBaseColumnSpec(BaseModel):
    """Base class for column specifications
    
    Derive from this and then put all the stuff you want to describe a column in
    it. There's a dictionary of these in the custom info.
    """

    pass

class FIBaseOptionalInfo(BaseModel):
    """Base placeholder class for optional info
    
    Not strictly necessary to do this but keeping it clean.
    """

    pass
