
from typing import List
from pydantic import BaseModel, Field


class FileRole(BaseModel):
    """Defines the structure for describing a single file's role."""
    file_path: str = Field(..., description="The relative path of the file.")
    role: str = Field(..., description="A brief description of the file's role within the package.")

class PackageDescriptionOutput(BaseModel):
    """Defines the overall expected JSON output structure for the task."""
    package_description: str = Field(..., description="A concise description of the package's overall purpose.")
    file_roles: List[FileRole] = Field(..., description="A list detailing the role of each file in the package.")

# We expect a dictionary where keys are package names (strings)
# and values are the refined description strings.
class RefinedPackageDescriptionOutput(BaseModel):
    package_id: str = Field(..., description="The unique identifier for the package.")
    package_description: str = Field(..., description="A concise description of the package's overall purpose.")

class RefinedDescriptionsOutput(BaseModel):
    """Defines the structure for describing a list of work packages."""
    package_descriptions: List[RefinedPackageDescriptionOutput] = Field(..., description="A list of mappings of package names to their refined descriptions.")
