import pandas as pd
from .knockout_sgRNA_selection_constant import *
from .logic import BaseState, Result_ProcessUserInput, BaseUserInputState
from llm import OpenAIChat
from .off_target import OffTarget
from .apis.parse_sgRNA_library import extract_info, sgRNA_library_reader
import time
 

class SgRNASelection(BaseUserInputState):
    prompt_process = PROMPT_PROCESS_AGENT1
    request_message = PROMPT_REQUEST_AGENT1

    @classmethod
    def step(cls, user_message, **kwargs):
        prompt = cls.prompt_process.format(user_message=user_message)
        response = OpenAIChat.chat(prompt, use_GPT4=True)
        response["original_request"] = user_message
        if response["Specified"].lower() == "yes":
            return (
                Result_ProcessUserInput(
                    status="success",
                    result=response,
                ),
                StateStepOriginal,
            )
        else:
            text_response = "Sugessted exon/exons: " + str(
                {
                    "Exon/exons": response["target exon"],
                    "Reason": {response["rationale"]},
                }
            )
            return (
                Result_ProcessUserInput(
                    status="success",
                    result=response,
                    response=text_response,
                ),
                StateStepQuestion,
            )


class StateStepQuestion(BaseUserInputState):
    prompt_process = PROMPT_PROCESS_QUESTION
    request_message = PROMPT_REQUEST_QUESTION
    prev_state_name = "SgRNASelection"

    @classmethod
    def step(cls, user_message, **kwargs):
        prev_result = kwargs["memory"][cls.prev_state_name].result
        target_exon = prev_result["target exon"]
        rationale = prev_result["rationale"]
        prompt = cls.prompt_process.format(
            user_message=user_message, target_exon=target_exon, rationale=rationale
        )
        response = OpenAIChat.chat(prompt, use_GPT4=True)
        choice = response["Choice"]
        if choice.lower() == "yes":
            return (
                Result_ProcessUserInput(
                    status="success", thoughts=response["Thoughts"], result=choice
                ),
                StateStepReformatted,
            )
        else:
            return (
                Result_ProcessUserInput(
                    status="success", thoughts=response["Thoughts"], result=choice
                ),
                StateStepOriginal,
            )


class StateStepReformatted(BaseState):
    prompt_process = PROMPT_PROCESS_AGENT2
    prev_state_name = "SgRNASelection"

    @classmethod
    def step(cls, user_message, **kwargs):
        start_time = time.time()
        response = kwargs["memory"][cls.prev_state_name].result
        if kwargs["is_automation"]:
            library = kwargs["memory"]["StateStep1Easy"].result
        else:
            library = kwargs["memory"]["StateStep1Inquiry"].result
        species = response["Species"]
        df = sgRNA_library_reader.parse_knockout_library(library, species)
        result_df, download_link = extract_info(
            response["reformatted_request"], cls.prompt_process, df
        )
        print(result_df)
        end_time = time.time()
        print("Time taken: ", end_time - start_time)
        if result_df.empty:
            return (
                Result_ProcessUserInput(
                    response=RESPONSE_STEP_ERROR,
                ),
                OffTargetQuestion,
            )
        return (
            Result_ProcessUserInput(
                status="success",
                result=result_df,
                response=download_link
                + "\n\n"
                + result_df[
                    [
                        "Target Gene Symbol",
                        "CRISPR Mechanism",
                        "sgRNA Sequence",
                        "PAM Sequence",
                        "Exon Number",
                        "sgRNA Cut Position (1-based)",
                        "On-Target Efficacy Score",
                        "Off-Target Rank",
                        "Combined Rank",
                    ]
                ].to_markdown(index=False),
            ),
            OffTargetQuestion,
        )


class StateStepOriginal(BaseState):
    prompt_process = PROMPT_PROCESS_AGENT2
    prev_state_name = "SgRNASelection"

    @classmethod
    def step(cls, user_message, **kwargs):
        response = kwargs["memory"][cls.prev_state_name].result
        if kwargs["is_automation"]:
            library = kwargs["memory"]["StateStep1Easy"].result
        else:
            library = kwargs["memory"]["StateStep1Inquiry"].result
        species = response["Species"]
        df = sgRNA_library_reader.parse_knockout_library(library, species)
        result_df, download_link = extract_info(
            response["original_request"], cls.prompt_process, df
        )
        if result_df.empty:
            return (
                Result_ProcessUserInput(
                    response=RESPONSE_STEP_ERROR,
                ),
                OffTargetQuestion,
            )
        return (
            Result_ProcessUserInput(
                status="success",
                result=result_df,
                response=download_link
                + "\n\n"
                + result_df[
                    [
                        "Target Gene Symbol",
                        "CRISPR Mechanism",
                        "sgRNA Sequence",
                        "PAM Sequence",
                        "Exon Number",
                        "sgRNA Cut Position (1-based)",
                        "On-Target Efficacy Score",
                        "Off-Target Rank",
                        "Combined Rank",
                    ]
                ].to_markdown(index=False),
            ),
            OffTargetQuestion,
        )


class OffTargetQuestion(BaseUserInputState):
    prompt_process = PROMPT_PROCESS_OFFTARGET_QUESTION
    request_message = PROMPT_REQUEST_OFFTARGET_QUESTION

    @classmethod
    def NextState(cls):
        return OffTarget

    @classmethod
    def step(cls, user_message, **kwargs):
        prompt = cls.prompt_process.format(user_message=user_message)
        response = OpenAIChat.chat(prompt, use_GPT4=True)

        result = response["Choice"]
        if result.lower() == "yes":
            return (
                Result_ProcessUserInput(
                    status="success", thoughts=response["Thoughts"], result=result
                ),
                cls.NextState(),
            )
        else:
            return (
                Result_ProcessUserInput(
                    status="success", thoughts=response["Thoughts"], result=result
                ),
                None,
            )