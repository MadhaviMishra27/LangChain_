from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):
    name:str
    age:int 
    state:str='UP' #passing default values
    email: Optional[EmailStr]=None
    cgpa: float= Field(gt=0, lt=10,default=8.5) #passing constraints and default value, we can also add a description here
    #we can also attach a regular expression to follow a pattern or something 
new_student={'name':'madhavi','age':22,'email':'madhavi@gmail.com','cgpa':8.25}
student=Student(**new_student)  #dictionary
print(student)
print(student.name)
student_dict=dict(student)
student_json=student.model_dump_json()