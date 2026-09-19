import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ============================================================
# VALIDATE API KEY
# ============================================================

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY was not found.\n\n"
        "Please create a .env file in the project root "
        "and add:\n\n"
        "GROQ_API_KEY=your_api_key"
    )


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# GET LLM RESPONSE
# ============================================================

def get_llm_response(
    prompt,
    temperature=0.1,
    max_tokens=8000,
    response_format=None
):
    """
    Send a prompt to Groq.

    Parameters:
        prompt:
            Prompt sent to the model.

        temperature:
            Controls randomness.

        max_tokens:
            Maximum output tokens.

        response_format:
            Optional Groq response format.

    Returns:
        Generated response as a string.
    """

    request_parameters = {
        "model": GROQ_MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": temperature,

        "max_completion_tokens": max_tokens
    }


    # --------------------------------------------------------
    # Add structured response format if provided
    # --------------------------------------------------------

    if response_format is not None:

        request_parameters[
            "response_format"
        ] = response_format


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            **request_parameters
        )

    except Exception as error:

        raise RuntimeError(
            f"Groq API request failed.\n\n"
            f"Model: {GROQ_MODEL}\n"
            f"Error: {error}"
        ) from error


    # --------------------------------------------------------
    # Extract response
    # --------------------------------------------------------

    if not response.choices:

        raise RuntimeError(
            "Groq returned no response choices."
        )


    message = response.choices[0].message

    content = message.content


    if not content:

        raise RuntimeError(
            "Groq returned an empty response."
        )


    return content

# import os

# from dotenv import load_dotenv
# from groq import Groq


# # ============================================================
# # LOAD ENVIRONMENT VARIABLES
# # ============================================================

# load_dotenv()


# # ============================================================
# # GROQ CONFIGURATION
# # ============================================================

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# GROQ_MODEL = os.getenv(
#     "GROQ_MODEL",
#     "openai/gpt-oss-120b"
# )


# # ============================================================
# # VALIDATE API KEY
# # ============================================================

# if not GROQ_API_KEY:

#     raise ValueError(
#         "GROQ_API_KEY was not found.\n\n"
#         "Please create a .env file in the project root "
#         "and add:\n\n"
#         "GROQ_API_KEY=your_api_key"
#     )


# # ============================================================
# # CREATE GROQ CLIENT
# # ============================================================

# client = Groq(
#     api_key=GROQ_API_KEY
# )


# # ============================================================
# # GET LLM RESPONSE
# # ============================================================

# def get_llm_response(
#     prompt,
#     temperature=0.2,
#     max_tokens=2000
# ):
#     """
#     Send a prompt to Groq and return the generated response.

#     Parameters:
#         prompt: Prompt to send to the LLM.
#         temperature: Controls randomness.
#         max_tokens: Maximum number of output tokens.

#     Returns:
#         Generated text.
#     """

#     try:

#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens
#         )

#         return response.choices[0].message.content

#     except Exception as error:

#         raise RuntimeError(
#             f"Groq API request failed.\n\n"
#             f"Model being used: {GROQ_MODEL}\n"
#             f"Error: {error}"
#         ) from error