"""
Parallel LLM Request Handler

Reuses libraries from AIAgent.py and utils/llm_client.py to send
parallel requests to multiple Ollama models.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Import config
try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("Config module not available")

# Import LLM client
from utils.llm_client import SimpleOllamaClient

# Import session logger
from .session_logger import SessionLogger

# Import response parser
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


@dataclass
class ParallelLLMResponse:
    """Response from a single model."""
    model: str
    response: str
    success: bool
    error: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    parsed_json: Optional[Dict[str, Any]] = None
    parsed_json_path: Optional[str] = None


@dataclass
class ParallelLLMRequest:
    """
    Parallel LLM request handler.
    
    Sends requests to multiple models in parallel and aggregates results.
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        parallel: Optional[bool] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize parallel LLM request handler.
        
        Args:
            models: List of model names. If None, uses OLLAMA_VERIFICATION_MODEL or OLLAMA_MODEL from config.
            base_url: Ollama base URL. If None, uses OLLAMA_BASE_URL from config.
            api_key: API key. If None, uses OLLAMA_API_KEY from config.
            timeout: Request timeout in seconds. If None, uses OLLAMA_VERIFICATION_TIMEOUT from config.
            parallel: Enable parallel requests. If None, uses OLLAMA_PARALLEL_REQUESTS from config.
            session_id: Optional session ID for logging. If None, generates a new one.
        """
        # Load config if available
        if CONFIG_AVAILABLE:
            llm_cfg = config.get_llm_config()
            
            if models is None:
                # Try verification_models first, then fallback to single model
                verification_models = llm_cfg.get('verification_models', [])
                if verification_models:
                    models = verification_models
                else:
                    verification_model = llm_cfg.get('verification_model')
                    if verification_model:
                        models = [verification_model]
                    else:
                        models = [llm_cfg.get('model', 'deepseek-r1:7b')]
            
            if base_url is None:
                base_url = llm_cfg.get('base_url', 'http://localhost:11434/v1')
            
            if api_key is None:
                api_key = llm_cfg.get('api_key', 'ollama')
            
            if timeout is None:
                timeout = llm_cfg.get('verification_timeout', 300)
            
            if parallel is None:
                parallel = llm_cfg.get('parallel_requests', True)
        else:
            # Defaults if config not available
            if models is None:
                models = ['deepseek-r1:7b']
            if base_url is None:
                base_url = 'http://localhost:11434/v1'
            if api_key is None:
                api_key = 'ollama'
            if timeout is None:
                timeout = 300
            if parallel is None:
                parallel = True
        
        self.models = models if isinstance(models, list) else [models] if models else []
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.parallel = parallel
        
        # Initialize session logger
        self.session_logger = SessionLogger(session_id=session_id)
        
        logger.info(f"ParallelLLMRequest initialized with {len(self.models)} models: {', '.join(self.models)}")
        logger.debug(f"Base URL: {base_url}, Timeout: {timeout}s, Parallel: {parallel}")
    
    def _create_client(self, model: str) -> Optional[SimpleOllamaClient]:
        """Create an Ollama client for the given model."""
        try:
            return SimpleOllamaClient(
                model=model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        except Exception as e:
            logger.warning(f"Failed to create client for model {model}: {e}")
            return None
    
    async def _send_request_async(
        self,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        request_type: str = "chat",
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> ParallelLLMResponse:
        """
        Send an async request to a single model.
        
        Args:
            model: Model name
            prompt: Prompt or messages
            request_type: Type of request
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            ParallelLLMResponse
        """
        client = self._create_client(model)
        if not client:
            return ParallelLLMResponse(
                model=model,
                response="",
                success=False,
                error="Failed to create client"
            )
        
        request_id = self.session_logger.log_request(
            model=model,
            prompt=prompt,
            request_type=request_type,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        
        try:
            response = await client.acomplete(
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Log response
            self.session_logger.log_response(
                model=model,
                response=response,
                request_id=request_id,
                request_type=request_type
            )
            
            # Parse response to JSON with metadata
            parsed_json = ResponseParser.parse(response, request_type)
            
            # Extract question text from prompt
            if isinstance(prompt, str):
                question_text = prompt
            elif isinstance(prompt, list) and len(prompt) > 0:
                # Extract content from message list
                if isinstance(prompt[0], dict):
                    question_text = prompt[0].get('content', str(prompt))
                else:
                    question_text = str(prompt[0])
            else:
                question_text = str(prompt)
            
            # Add metadata to parsed JSON
            parsed_json['metadata'] = {
                'question': question_text,
                'model': model,
                'date_time': datetime.now().isoformat(),
                'session': self.session_logger.session_id,
                'request_id': request_id,
                'request_type': request_type,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            # Save parsed JSON to file
            safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
            json_filename = f"parsed_{request_type}_{safe_model}_{request_id}.json"
            json_path = self.session_logger.log_json(
                data=parsed_json,
                filename=json_filename,
                subfolder="parsed"
            )
            
            return ParallelLLMResponse(
                model=model,
                response=response,
                success=True,
                request_id=request_id,
                parsed_json=parsed_json,
                parsed_json_path=json_path
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Request failed for model {model}: {error_msg}")
            
            # Log error response
            self.session_logger.log_response(
                model=model,
                response=f"ERROR: {error_msg}",
                request_id=request_id,
                request_type=request_type,
                metadata={'error': error_msg}
            )
            
            return ParallelLLMResponse(
                model=model,
                response="",
                success=False,
                error=error_msg,
                request_id=request_id
            )
    
    async def send_parallel_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        request_type: str = "chat",
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> List[ParallelLLMResponse]:
        """
        Send requests to all models in parallel (async).
        
        Args:
            prompt: Prompt or messages
            request_type: Type of request
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of ParallelLLMResponse
        """
        if not self.models:
            logger.warning("No models configured")
            return []
        
        logger.info(f"Sending parallel requests to {len(self.models)} models: {', '.join(self.models)}")
        
        # Create tasks for all models
        tasks = [
            self._send_request_async(model, prompt, request_type, temperature, max_tokens)
            for model in self.models
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model = self.models[i]
                logger.warning(f"Exception for model {model}: {result}")
                responses.append(ParallelLLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        # Log summary
        successful = [r for r in responses if r.success]
        failed = [r for r in responses if not r.success]
        
        logger.info(f"Parallel requests completed: {len(successful)} successful, {len(failed)} failed")
        
        # Log summary JSON
        summary = {
            'session_id': self.session_logger.session_id,
            'total_models': len(self.models),
            'successful': len(successful),
            'failed': len(failed),
            'models': {
                'all': self.models,
                'successful': [r.model for r in successful],
                'failed': [r.model for r in failed]
            },
            'responses': [
                {
                    'model': r.model,
                    'success': r.success,
                    'error': r.error,
                    'response_length': len(r.response) if r.response else 0
                }
                for r in responses
            ]
        }
        self.session_logger.log_json(summary, 'parallel_request_summary.json')
        
        return responses
    
    def send_parallel(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        request_type: str = "chat",
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> List[ParallelLLMResponse]:
        """
        Send requests to all models in parallel (synchronous wrapper).
        
        Args:
            prompt: Prompt or messages
            request_type: Type of request
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of ParallelLLMResponse
        """
        if not self.parallel or len(self.models) == 1:
            # Sequential fallback
            return self.send_sequential(prompt, request_type, temperature, max_tokens)
        
        # Use async parallel execution
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not available - may encounter event loop issues")
        
        try:
            return asyncio.run(self.send_parallel_async(prompt, request_type, temperature, max_tokens))
        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}, falling back to sequential")
            return self.send_sequential(prompt, request_type, temperature, max_tokens)
    
    def send_sequential(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        request_type: str = "chat",
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> List[ParallelLLMResponse]:
        """
        Send requests to all models sequentially.
        
        Args:
            prompt: Prompt or messages
            request_type: Type of request
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of ParallelLLMResponse
        """
        if not self.models:
            logger.warning("No models configured")
            return []
        
        logger.info(f"Sending sequential requests to {len(self.models)} models")
        
        responses = []
        for model in self.models:
            client = self._create_client(model)
            if not client:
                responses.append(ParallelLLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error="Failed to create client"
                ))
                continue
            
            request_id = self.session_logger.log_request(
                model=model,
                prompt=prompt,
                request_type=request_type,
                metadata={
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            )
            
            try:
                response = client.complete(
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                self.session_logger.log_response(
                    model=model,
                    response=response,
                    request_id=request_id,
                    request_type=request_type
                )
                
                # Parse response to JSON with metadata
                parsed_json = ResponseParser.parse(response, request_type)
                
                # Extract question text from prompt
                if isinstance(prompt, str):
                    question_text = prompt
                elif isinstance(prompt, list) and len(prompt) > 0:
                    # Extract content from message list
                    if isinstance(prompt[0], dict):
                        question_text = prompt[0].get('content', str(prompt))
                    else:
                        question_text = str(prompt[0])
                else:
                    question_text = str(prompt)
                
                # Add metadata to parsed JSON
                parsed_json['metadata'] = {
                    'question': question_text,
                    'model': model,
                    'date_time': datetime.now().isoformat(),
                    'session': self.session_logger.session_id,
                    'request_id': request_id,
                    'request_type': request_type,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
                
                # Save parsed JSON to file
                safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
                json_filename = f"parsed_{request_type}_{safe_model}_{request_id}.json"
                json_path = self.session_logger.log_json(
                    data=parsed_json,
                    filename=json_filename,
                    subfolder="parsed"
                )
                
                responses.append(ParallelLLMResponse(
                    model=model,
                    response=response,
                    success=True,
                    request_id=request_id,
                    parsed_json=parsed_json,
                    parsed_json_path=json_path
                ))
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Request failed for model {model}: {error_msg}")
                
                self.session_logger.log_response(
                    model=model,
                    response=f"ERROR: {error_msg}",
                    request_id=request_id,
                    request_type=request_type,
                    metadata={'error': error_msg}
                )
                
                responses.append(ParallelLLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error=error_msg,
                    request_id=request_id
                ))
        
        return responses
    
    def get_session_id(self) -> str:
        """Get the session ID."""
        return self.session_logger.session_id
    
    def get_session_dir(self) -> str:
        """Get the session logs directory."""
        return self.session_logger.get_session_dir()

