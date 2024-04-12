from enum import Enum

class EditType(Enum):
    AddKnowledge = 1
    EditSchema = 2
    UpdateSamples = 3



class EditSamplesAction(Enum):
    Add = 1
    Remove = 2