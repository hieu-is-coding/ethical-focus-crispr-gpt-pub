from dataclasses import dataclass, field
import numbers
from typing import Optional, List, Any, Dict
from llm import OpenAIChat
from .safety import (
    check_human_heritable_editing,
    WARNING_HUMAN_HERITABLE_EDITING,
    check_stopping_keywords,
    contains_identifiable_genes,
    WARNING_PRIVACY,
    _check_contains_keyword_list,
    SemanticSafetyFilter,
)
from util import get_logger
import json

logger = get_logger(__name__)

_semantic_filter = SemanticSafetyFilter()

@dataclass
class Result_ProcessUserInput:
    status: str = "success"
    result: Optional[str] = None
    thoughts: Optional[str] = None
    response: Optional[str] = None


# ============================================================================
# AI SAFETY AGENT USING GEMINI/GPT WITH TOOL CALLING
# ============================================================================

class AISafetyAgent:
    """
    AI Safety Agent that uses existing safety.py functions as tools.
    Provides intelligent assessment beyond simple keyword matching.
    """
    
    SYSTEM_PROMPT = """You are an expert biosafety and bioethics AI assistant for CRISPR-GPT system.

Your role is to evaluate user requests for CRISPR gene editing and determine if they are safe to process.

You have access to the following safety check tools:
1. check_stopping_keywords - Checks for dangerous keywords (pathogens, germline editing, etc.)
2. contains_identifiable_genes - Checks for human genetic sequences that may violate privacy
3. check_human_heritable_editing - Checks for human germline/heritable editing indicators
4. check_keyword_list - Custom keyword list checking

EVALUATION PROCESS:
1. First, use the available tools to scan the request
2. Analyze the intent and context beyond just keywords
3. Consider:
   - Biosecurity risks (pathogen work, gain-of-function, dual-use)
   - Ethical concerns (germline editing, eugenics, discrimination)
   - Privacy violations (patient sequences, identifiable data)
   - Hidden intent or evasion attempts
4. Make a final safety determination

RESPONSE FORMAT (JSON):
{
    "risk_level": "safe|low|medium|high|critical",
    "confidence": 0.0-1.0,
    "tool_results": {
        "stopping_keywords": "ok or error message",
        "identifiable_genes": true/false,
        "heritable_editing": true/false,
        "semantic_safety": unsafe/safe,
    },
    "analysis": "Your detailed analysis beyond tool results",
    "red_flags": ["flag1", "flag2"],
    "decision": "allow|warn|block",
    "user_message": "Message to show the user"
}

IMPORTANT:
- Be thorough but not overly restrictive
- Consider legitimate research use cases
- If tools flag issues, analyze if they're false positives
- Provide clear explanations for your decisions
- Default to caution when uncertain"""

    def __init__(self):
        self.call_count = 0
    
    def evaluate_request(self, user_message: str, memory: Dict) -> Dict:
        """
        Evaluate user request using AI agent with safety tools.
        
        Args:
            user_message: The user's input
            memory: Session memory containing history and context
            
        Returns:
            Dict with risk_level, decision, and user_message
        """
        self.call_count += 1
        
        # Step 1: Run safety tools
        tool_results = self._run_safety_tools(user_message)
        
        # Step 2: Build context
        session_history = memory.get('message_history', [])[-5:]  # Last 5 messages
        previous_flags = memory.get('safety_flags', [])
        
        # Step 3: Prepare prompt for AI agent
        prompt = f"""{self.SYSTEM_PROMPT}

=== CURRENT REQUEST ===
"{user_message}"

=== TOOL RESULTS ===
{json.dumps(tool_results, indent=2)}

=== SESSION CONTEXT ===
Recent History (last 5 requests): {json.dumps(session_history, indent=2)}
Previous Safety Flags: {json.dumps(previous_flags, indent=2)}

=== YOUR TASK ===
Analyze this request comprehensively and provide your safety assessment in JSON format.
Consider:
1. What the tools found
2. The user's intent and context
3. Patterns in their behavior
4. Potential risks not caught by tools
5. Whether this is legitimate research

Provide your response in valid JSON format matching the schema above."""

        try:
            response = OpenAIChat.chat(prompt, use_GPT4=True)
            print("-----------  $    response: ", response)
            
            # Validate response
            if not isinstance(response, dict):
                logger.error(f"AI agent returned non-dict: {response}")
                return self._create_fallback_assessment(tool_results)
            
            # Ensure required fields
            if 'decision' not in response:
                response['decision'] = self._infer_decision(response.get('risk_level', 'high'))
            
            if 'user_message' not in response:
                response['user_message'] = self._generate_user_message(response)
            
            # Log assessment
            logger.info(f"AI Safety Agent Assessment: {response.get('risk_level')} - {response.get('decision')}")
            
            return response
            
        except Exception as e:
            logger.error(f"AI Safety Agent error: {e}")
            # Fallback to tool-based assessment
            return self._create_fallback_assessment(tool_results)
    
    def _run_safety_tools(self, user_message: str) -> Dict:
        """Run all safety check tools from safety.py"""
        
        results = {}
        
        # Handle None or empty messages
        if not user_message or not isinstance(user_message, str):
            return {
                'stopping_keywords': 'ok',
                'identifiable_genes': False,
                'heritable_editing': False,
                'note': 'Empty or invalid message - skipped checks'
            }
        
        # Tool 1: Check stopping keywords
        try:
            keyword_result = check_stopping_keywords(user_message)
            results['stopping_keywords'] = keyword_result
        except Exception as e:
            logger.error(f"Tool error (stopping_keywords): {e}")
            results['stopping_keywords'] = 'ok'  # Fail-safe to allow on error
        
        # Tool 2: Check for identifiable genes
        try:
            results['identifiable_genes'] = contains_identifiable_genes(user_message)
        except Exception as e:
            logger.error(f"Tool error (identifiable_genes): {e}")
            results['identifiable_genes'] = False
        
        # Tool 3: Check human heritable editing
        try:
            results['heritable_editing'] = check_human_heritable_editing(user_message)
        except Exception as e:
            logger.error(f"Tool error (heritable_editing): {e}")
            results['heritable_editing'] = False
        
        # Tool 4: Check semantic safety
        try:
            semantic_result = _semantic_filter.check_semantic_safety(user_message)
            results['semantic_safety'] = semantic_result
        except Exception as e:
            logger.error(f"Semantic safety error: {e}")
            results['semantic_safety'] = {"unsafe": False}

        return results
    
    def _infer_decision(self, risk_level: str) -> str:
        """Infer decision from risk level"""
        if risk_level in ['critical', 'high']:
            return 'block'
        elif risk_level == 'medium':
            return 'warn'
        else:
            return 'allow'
    
    def _generate_user_message(self, assessment: Dict) -> str:
        """Generate user-facing message from assessment"""
        
        decision = assessment.get('decision', 'block')
        risk_level = assessment.get('risk_level', 'unknown')
        analysis = assessment.get('analysis', 'Safety check completed.')
        red_flags = assessment.get('red_flags', [])
        
        if decision == 'block':
            msg = f"🛑 **Safety Check: Request Blocked ({risk_level.upper()})**\n\n"
            msg += f"{analysis}\n\n"
            if red_flags:
                msg += "**Specific Concerns:**\n"
                for flag in red_flags:
                    msg += f"  • {flag}\n"
            return msg
        
        elif decision == 'warn':
            msg = f"⚠️ **Safety Advisory ({risk_level.upper()})**\n\n"
            msg += f"{analysis}\n\n"
            if red_flags:
                msg += "**Please Note:**\n"
                for flag in red_flags:
                    msg += f"  • {flag}\n"
            msg += "\nYou may proceed, but please review these considerations carefully."
            return msg
        
        else:  # allow
            return ""  # No message needed for safe requests
    
    def _create_fallback_assessment(self, tool_results: Dict) -> Dict:
        """Create assessment based purely on tool results when AI agent fails"""
        
        # Check if any tools flagged issues
        keyword_result = tool_results.get('stopping_keywords', 'ok')
        has_genes = tool_results.get('identifiable_genes', False)
        has_heritable = tool_results.get('heritable_editing', False)
        semantic = tool_results.get('semantic_safety', {"unsafe": False})
        
        if keyword_result != 'ok':
            return {
                'risk_level': 'critical',
                'decision': 'block',
                'user_message': keyword_result,
                'tool_results': tool_results,
                'analysis': 'Blocked by keyword filter',
                'red_flags': ['Dangerous keyword detected'],
                'confidence': 1.0
            }
        
        if has_genes:
            return {
                'risk_level': 'high',
                'decision': 'block',
                'user_message': WARNING_PRIVACY,
                'tool_results': tool_results,
                'analysis': 'Privacy violation detected',
                'red_flags': ['Identifiable genetic sequence detected'],
                'confidence': 0.9
            }
        
        if semantic.get("unsafe"):
            return {
                "risk_level": "high",
                "decision": "block",
                "user_message": f"Semantic filter detected potential unsafe intent related to {semantic.get('label')}.",
                "tool_results": tool_results,
                "analysis": f"User message semantically matched unsafe term ({semantic.get('label')})",
                "red_flags": [semantic.get('matched_text')],
                "confidence": 0.9
            }
        
        if has_heritable:
            return {
                'risk_level': 'medium',
                'decision': 'warn',
                'user_message': WARNING_HUMAN_HERITABLE_EDITING,
                'tool_results': tool_results,
                'analysis': 'Human heritable editing detected',
                'red_flags': ['Human germline/heritable editing indicators'],
                'confidence': 0.8
            }
        
        # No issues found
        return {
            'risk_level': 'safe',
            'decision': 'allow',
            'user_message': '',
            'tool_results': tool_results,
            'analysis': 'No safety concerns detected',
            'red_flags': [],
            'confidence': 0.7
        }


# Global AI safety agent instance
_ai_safety_agent = None

def get_ai_safety_agent() -> AISafetyAgent:
    """Get or create the global AI safety agent instance"""
    global _ai_safety_agent
    if _ai_safety_agent is None:
        _ai_safety_agent = AISafetyAgent()
    return _ai_safety_agent


# ============================================================================
# ORIGINAL CLASSES WITH ENHANCED SAFETY
# ============================================================================

class BaseState:
    isFinal = False
    request_user_input = False
    request_message = ""

    @classmethod
    def FallbackState(cls):
        return None

    @classmethod
    def get_request_message(cls):
        return cls.request_message

    @classmethod
    def step(cls, user_message, **kwargs):
        if not cls.isFinal:
            raise NotImplementedError
        return Result_ProcessUserInput(), cls

    @classmethod
    def safe_step(cls, user_message, **kwargs):
        try:
            if user_message is not None:
                logger.debug("DEBUG: there might be some error in the code.")
            return cls.step(user_message, **kwargs)
        except Exception as ex:
            logger.info(["Error occured", ex])
            return (
                Result_ProcessUserInput(
                    status="error",
                    response="Error occured. Error Message: "
                    + str(ex)
                    + " Let's try again.",
                ),
                cls.FallbackState(),
            )


class BaseUserInputState:
    isFinal = False
    request_user_input = True
    prompt_process = "{user_messsage}"
    request_message = ""

    @classmethod
    def NextState(cls):
        return cls

    @classmethod
    def get_request_message(cls):
        return cls.request_message

    @classmethod
    def step(cls, user_message, **kwargs):
        prompt = cls.prompt_process.format(user_message=user_message)
        response = OpenAIChat.chat(prompt)
        return (
            Result_ProcessUserInput(
                status="success",
                thoughts=response["Thoughts"],
                result=response["Choice"],
                response=str(response),
            ),
            cls.NextState(),
        )

    @classmethod
    def safe_step(cls, user_message, **kwargs):
        try:
            # Handle None or empty messages - skip safety check for system messages
            if user_message is None or not isinstance(user_message, str):
                return cls.step(user_message, **kwargs)
            
            # Initialize message history if not exists
            if 'message_history' not in kwargs.get('memory', {}):
                kwargs.setdefault('memory', {})['message_history'] = []
            
            if(user_message.isdigit()):
                return cls.step(user_message, **kwargs)

            # Use AI Safety Agent for comprehensive evaluation
            safety_agent = get_ai_safety_agent()
            assessment = safety_agent.evaluate_request(
                user_message, 
                kwargs.get('memory', {})
            )
            
            # Log assessment details
            logger.info(f"Safety Assessment: {assessment.get('risk_level')} - {assessment.get('decision')}")
            
            # Handle different decisions
            decision = assessment.get('decision', 'block')
            user_msg = assessment.get('user_message', '')
            
            # Special handling for confirmation messages
            if user_message.strip().lower() in ['confirm', 'yes', 'y', 'ok', 'proceed']:
                logger.info(f"User confirmation received: {user_message}")
                # Allow confirmation to proceed without additional warnings
                decision = 'allow'
                user_msg = ''
            
            if decision == 'block':
                # Store safety flag
                kwargs['memory'].setdefault('safety_flags', []).append({
                    'message': user_message[:100],
                    'risk_level': assessment.get('risk_level'),
                    'red_flags': assessment.get('red_flags', [])
                })
                
                return (
                    Result_ProcessUserInput(status="error", response=user_msg),
                    cls,
                )
            
            elif decision == 'warn':
                # For human heritable editing, use the existing ACK mechanism
                # if assessment.get('tool_results', {}).get('heritable_editing'):
                #     if not kwargs["memory"].get("flag_human_heritable_editing_ack", False):
                #         kwargs["memory"]["flag_human_heritable_editing_ack"] = True
                #         kwargs["memory"]["cached_user_message_before_ack"] = user_message
                #         return Result_ProcessUserInput(
                #             status="error", response=user_msg
                #         ), make_check_ack_state(cls)
                
                # For other warnings, show message but continue
                # if user_message.startswith("Q:"):
                #     qa_result = OpenAIChat.QA(user_message, use_GPT4=True)
                #     response_text = user_msg + "\n\n" + qa_result if user_msg else qa_result
                #     return Result_ProcessUserInput(response=response_text), cls
                # else:
                # Continue with normal processing but prepend warning
                result, next_state = cls.step(user_message, **kwargs)
                if user_msg and result.response:
                    result.response = user_msg + "\n\n" + result.response
                elif user_msg:
                    result.response = user_msg
                
                # Store in history
                kwargs['memory']['message_history'].append(user_message)
                return result, next_state
            
            else:  # decision == 'allow'
                # Process normally
                # Handle Q: prefix before processing
                if user_message.startswith("Q:"):
                    qa_result = OpenAIChat.QA(user_message, use_GPT4=True)
                    return Result_ProcessUserInput(response=qa_result), cls
                
                # Store in history
                kwargs['memory']['message_history'].append(user_message)
                return cls.step(user_message, **kwargs)
                    
        except Exception as ex:
            logger.info(["Error occured", ex])
            return (
                Result_ProcessUserInput(
                    status="error",
                    response="Error occured. Error Message: "
                    + str(ex)
                    + " Let's try again.",
                ),
                cls,
            )


class gradio_state_machine:
    """ For automation mode only."""
    def __init__(self, task_list):
        self.MAX_ITER = 100
        self.full_task_list = task_list
        self.reset()

    def reset(self):
        self.todo_task_list = self.full_task_list[:]
        self.current_state = self.todo_task_list.pop(0)
        self.memory = dict()
        self.cached_message = []
        self.state_stack = []

    def append_message(self, s):
        self.cached_message.append(s)
        # self.cached_message += '\n\n'

    def clear_message(self):
        display = self.cached_message
        self.cached_message = []
        return display

    def loop(self, user_message, email="", files=[]):
        for _ in range(self.MAX_ITER):
            response, next_state = self.current_state.safe_step(
                user_message,
                memory=self.memory,
                email=email,
                files=files,
                is_automation=True,
            )
            # logger.info(self.current_state.__name__)
            self.memory[self.current_state.__name__] = response
            _from_ack_state = self.current_state.__name__ == "StateCheckACK"

            if response.response is not None:
                self.append_message(response.response)
            self.state_stack.append(self.current_state)
            if next_state is None:  # finish a subtask, fetch the next one
                self.current_state = self.todo_task_list.pop(0)
            elif isinstance(
                next_state, list
            ):  ## include a list of entry state of each subtask
                self.todo_task_list.extend(next_state)
                self.current_state = self.todo_task_list.pop(0)
            else:
                self.current_state = (
                    next_state  # continue to next state within the same subtask.
                )
            request_msg = self.current_state.get_request_message()
            if response.status != "error" and len(request_msg) > 0:
                self.append_message(request_msg)

            if self.current_state.isFinal:
                return self.clear_message()  # flush output and wait for next input.
            if self.current_state.request_user_input:
                ## special rule: if returned from checkAck state, then fetch user input from cache
                if _from_ack_state:
                    user_message = self.memory.get("cached_user_message_before_ack")
                    self.memory["cached_user_message_before_ack"] = None
                else:
                    return self.clear_message()  # flush output and wait for next input.
            else:
                user_message = None


@dataclass
class GradioMachineStateClass:
    full_task_list: Optional[List] = None
    todo_task_list: Optional[List] = None
    current_state: Optional[Any] = None
    memory: Optional[Dict] = field(default_factory=dict)
    cached_message: Optional[List] = field(default_factory=list)
    state_stack: Optional[List] = field(default_factory=list)


class concurrent_gradio_state_machine:
    MAX_ITER = 100
    """
        Use Gradio.State to manage states within sessions.
    """

    # def __init__(cls, task_list):
    #     cls.full_task_list = task_list
    #     cls.reset()
    @classmethod
    def reset(cls, mystate):
        mystate.todo_task_list = mystate.full_task_list[:]
        mystate.current_state = mystate.todo_task_list.pop(0)
        mystate.memory = dict()
        mystate.cached_message = []
        mystate.state_stack = []

    @classmethod
    def append_message(cls, s, mystate):
        mystate.cached_message.append(s)
        # cls.cached_message += '\n\n'

    @classmethod
    def clear_message(cls, mystate):
        display = mystate.cached_message
        mystate.cached_message = []
        return display

    @classmethod
    def loop(cls, user_message, mystate, email="", files=[]):
        for _ in range(cls.MAX_ITER):
            response, next_state = mystate.current_state.safe_step(
                user_message=user_message,
                memory=mystate.memory,
                email=email,
                files=files,
                is_automation=False,
            )
            # logger.info(mystate.current_state.__name__)
            mystate.memory[mystate.current_state.__name__] = response
            _from_ack_state = mystate.current_state.__name__ == "StateCheckACK"

            if response.response is not None:
                cls.append_message(response.response, mystate)
            mystate.state_stack.append(mystate.current_state)
            if next_state is None:  # finish a subtask, fetch the next one
                mystate.current_state = mystate.todo_task_list.pop(0)
            elif isinstance(
                next_state, list
            ):  ## include a list of entry state of each subtask
                mystate.todo_task_list.extend(next_state)
                mystate.current_state = mystate.todo_task_list.pop(0)
            else:
                mystate.current_state = (
                    next_state  # continue to next state within the same subtask.
                )
            request_msg = mystate.current_state.get_request_message()
            if response.status != "error" and len(request_msg) > 0:
                cls.append_message(request_msg, mystate)

            if mystate.current_state.isFinal:
                return cls.clear_message(
                    mystate
                )  # flush output and wait for next input.
            if mystate.current_state.request_user_input:
                ## special rule: if returned from checkAck state, then fetch user input from cache
                if _from_ack_state:
                    user_message = mystate.memory.get("cached_user_message_before_ack")
                    mystate.memory["cached_user_message_before_ack"] = None
                else:
                    return cls.clear_message(
                        mystate
                    )  # flush output and wait for next input.
            else:
                user_message = None


class StateFinal(BaseState):
    request_user_input = False
    isFinal = True
    request_message = "Finished. Clear the current chat or start a new one."


class EmptyState(BaseState):
    @classmethod
    def step(cls, user_message, **kwargs):
        return Result_ProcessUserInput(), None


class EmptyStateFinal(BaseState):
    request_user_input = False
    isFinal = True
    request_message = ""


class StateCheckACK(BaseUserInputState):
    def __init__(self, ret=None):
        self.ret = ret
        self.__name__ = "StateCheckACK"

    def safe_step(cls, user_message, **kwargs):
        try:
            # Handle None or empty messages - skip safety check for system messages
            if user_message is None or not isinstance(user_message, str):
                if user_message is None:
                    logger.debug("Received None message in StateCheckACK, skipping safety check")
                    return cls.step(user_message, **kwargs)
                
            if(user_message.isdigit()):
                return cls.step(user_message, **kwargs)
                
            # Use AI Safety Agent for ACK state as well
            safety_agent = get_ai_safety_agent()
            assessment = safety_agent.evaluate_request(
                user_message, 
                kwargs.get('memory', {})
            )
            
            decision = assessment.get('decision', 'block')
            user_msg = assessment.get('user_message', '')
            
            if decision == 'block':
                return (
                    Result_ProcessUserInput(status="error", response=user_msg),
                    cls,
                )
            elif user_message.startswith("Q:"):
                qa_result = OpenAIChat.QA(user_message, use_GPT4=True)
                return Result_ProcessUserInput(response=qa_result), cls
            else:
                return cls.step(user_message, **kwargs)
        except Exception as ex:
            logger.info(["Error occured", ex])
            return (
                Result_ProcessUserInput(
                    status="error",
                    response="Error occured. Error Message: "
                    + str(ex)
                    + " Let's try again.",
                ),
                cls,
            )

    def step(self, user_message, **kwargs):
        if user_message.lower() in ["y", "yes"]:
            return Result_ProcessUserInput(status="success"), self.ret
        else:
            return Result_ProcessUserInput(status="error"), self


def make_check_ack_state(ret):
    return StateCheckACK(ret)