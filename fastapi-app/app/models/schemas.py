from pydantic import BaseModel
from typing import Optional, List

class UserLogin(BaseModel):
    matricule: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    anonymous_id: str
    groups: List[str]

class TokenData(BaseModel):
    anonymous_id: Optional[str] = None
    groups: List[str] = []

class UserInfo(BaseModel):
    uid: str
    cn: str
    sn: Optional[str] = None
    givenName: Optional[str] = None
    mail: Optional[str] = None
    uidNumber: Optional[str] = None
    gidNumber: Optional[str] = None
    homeDirectory: Optional[str] = None
    loginShell: Optional[str] = None
    groups: List[str] = []