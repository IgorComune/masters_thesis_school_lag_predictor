from pydantic import BaseModel

class InferenceRequest(BaseModel):
    ipv: float
    ips: float
    iaa: float
    ieg: float
    no_av: float
    ida: float
