from typing import Annotated


def get_full_name(
    first_name: Annotated[str, "This is your first name parameter"],
    last_name: Annotated[str, "This is your last name parameter"],
):
    return first_name.title() + " " + last_name.title()


def get_name_with_age(name: str, age: int):
    return name.title() + " is this old: " + str(age)


print(get_full_name("muhammad", "attar"))
print(get_name_with_age("attar", 21))
