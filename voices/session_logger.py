"""
Session-based logging for LLM requests and responses.

Stores all logs in logs/sessionid/ subfolder for easy session tracking.
"""

import os
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Session-based logger for LLM requests and responses.
    
    Creates a subfolder in logs/ with session ID to store all related logs.
    """
    
    def __init__(self, session_id: Optional[str] = None, base_logs_dir: Optional[str] = None):
        """
        Initialize session logger.
        
        Args:
            session_id: Optional session ID. If None, generates a new one.
            base_logs_dir: Base logs directory. Defaults to logs/ in project root.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.session_id = session_id
        
        # Determine base logs directory
        if base_logs_dir is None:
            # Default to logs/ in project root (parent of voices/)
            base_logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        
        self.base_logs_dir = base_logs_dir
        self.session_logs_dir = os.path.join(base_logs_dir, session_id)
        
        # Create session directory
        os.makedirs(self.session_logs_dir, exist_ok=True)
        
        logger.info(f"SessionLogger initialized with session_id: {session_id}")
        logger.debug(f"Session logs directory: {self.session_logs_dir}")
    
    def log_request(
        self,
        model: str,
        prompt: str,
        request_type: str = "llm_request",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an LLM request.
        
        Args:
            model: Model name
            prompt: Request prompt/messages
            request_type: Type of request (e.g., "chat", "verification")
            metadata: Optional metadata to include
            
        Returns:
            Request ID for correlation
        """
        request_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Create safe filename
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        filename = f"request_{request_type}_{safe_model}_{timestamp}_{request_id}.txt"
        filepath = os.path.join(self.session_logs_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Request ID: {request_id}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Request Type: {request_type}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if metadata:
                    f.write(f"\nMetadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("PROMPT:\n")
                f.write("="*80 + "\n")
                
                # Handle both string and list of messages
                if isinstance(prompt, str):
                    f.write(prompt)
                elif isinstance(prompt, list):
                    for msg in prompt:
                        if isinstance(msg, dict):
                            f.write(f"{msg.get('role', 'user')}: {msg.get('content', '')}\n")
                        else:
                            f.write(f"{msg}\n")
                else:
                    f.write(str(prompt))
                
                f.write("\n" + "="*80 + "\n")
            
            logger.debug(f"Logged request to: {filepath}")
            return request_id
            
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")
            return request_id
    
    def log_response(
        self,
        model: str,
        response: str,
        request_id: str,
        request_type: str = "llm_request",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an LLM response.
        
        Args:
            model: Model name
            response: Response text
            request_id: Request ID for correlation
            request_type: Type of request
            metadata: Optional metadata to include
            
        Returns:
            Response file path
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Create safe filename
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        filename = f"response_{request_type}_{safe_model}_{timestamp}_{request_id}.txt"
        filepath = os.path.join(self.session_logs_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Request ID: {request_id}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Request Type: {request_type}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Response Length: {len(response) if response else 0}\n")
                
                if metadata:
                    f.write(f"\nMetadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("RESPONSE:\n")
                f.write("="*80 + "\n")
                f.write(response if response else "(empty response)")
                f.write("\n" + "="*80 + "\n")
            
            logger.debug(f"Logged response to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.warning(f"Failed to log response: {e}")
            return ""
    
    def log_json(
        self,
        data: Dict[str, Any],
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Log JSON data to session directory.
        
        Args:
            data: Data to log as JSON
            filename: Filename (can include timestamp prefix or not)
            subfolder: Optional subfolder within session directory (e.g., "parsed")
            
        Returns:
            File path
        """
        target_dir = self.session_logs_dir
        if subfolder:
            target_dir = os.path.join(target_dir, subfolder)
            os.makedirs(target_dir, exist_ok=True)
        
        # If filename doesn't already have timestamp, add one
        if not any(c.isdigit() for c in filename[:8]):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(target_dir, f"{timestamp}_{filename}")
        else:
            filepath = os.path.join(target_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Logged JSON to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.warning(f"Failed to log JSON: {e}")
            return ""
    
    def get_session_dir(self) -> str:
        """Get the session logs directory path."""
        return self.session_logs_dir

