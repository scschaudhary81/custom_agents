from pydantic import BaseModel,Field

class SearchResponseAgent(BaseModel):
    message: str = Field(description="Search response message")
    urls: list[str] = Field(description="List of URLS which were used to generate the search response")