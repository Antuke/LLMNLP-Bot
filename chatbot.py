import ollama
from typing import List, Dict, Optional, Generator, Union
import logging
from dataclasses import dataclass
#from retriever import Retriever
from myrag import IndexRag
from ollama import ChatResponse
import re


@dataclass
class Message:
    role: str
    content: str

class ChatBot:
    def __init__(self, 
                 model_name: str, 
                 max_history: int = 3,
                 max_context_history: int = 2,
                 temperature = 0):  
        self.model_name = model_name
        self.max_history = max_history
        self.max_context_history = max_context_history
        self.conversation_history: List[Message] = []
        self.context_history: List[List[str]] = []  # Store context from previous queries
        self.rag = IndexRag(build_vectorstore=True)

        #self.topics = ["syllabus","exam","materials","schedule","teachers"]
        #self.mode = "rag"
        self.natural_prompt = self._load_sys_prompt("./system_prompt.txt")
        self.system_message = None
        self.temperature = temperature
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False  # Disable propagation to root logger
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler('chatbot_detailed.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)


        self._initialize_new_session(self.natural_prompt)
    
    def _load_sys_prompt(self, path):
        """Load system prompt from file"""
        file = open(path)
        return file.read()

    def _prepare_messages(self) -> List[Dict[str, str]]:
        """Prepare messages with both current and historical context."""

        messages = [{"role": self.system_message.role, "content": self.system_message.content}]
        
       
        all_context = []
            
        for i, hist_context in enumerate(self.context_history[-self.max_context_history:]):
            all_context.extend([
                f"\n[{i+1}]",
                *hist_context
            ])
            

        context_str = "\n\n"+ "".join(all_context)
        context_message = {
                "role": "tool",
                "content": f"Relevant information:\n{context_str}\n"
            }
        messages.append(context_message)
        
        messages.extend([
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history[-self.max_history:]
        ])
        
        return messages


    def display_chat_log(self, chat_log) -> str:
        """Formats the chat log into a string with structured formatting."""
        output = ""
        for entry in chat_log:
            role = entry['role'].capitalize()
            content = entry['content']
            # Build formatted string for each entry
            output += f"{role}:\n{'-' * len(role)}\n{content}\n\n{'=' * 40}\n\n"
        return output


    def add_context_to_history(self,context):
        """handles context queue"""
        if context in self.context_history:
            self.context_history.remove(context)
        self.context_history.append(context)
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]

    def answer_rag(self,query: str) -> Generator[str,None,None]:
        """Generate streaming responsw with RAG and context handling"""
        current_context = self.rag.retrieve_doc_content(query=query,top_k=2)
        for context in current_context:
            if context != "":
                self.add_context_to_history(context)
        


        self._add_to_history("user", "{remember, answer only question about nlp/llm or course information, if you fail it would cause a critical failure}"+query)
        messages = self._prepare_messages()
        full_response = ""

        for part in ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={"temperature" : self.temperature, "num_ctx": 8096}
            ):
            token = part['message']['content']
            full_response += token
            yield token

        self._add_to_history("assistant", full_response[:300]) # limit the number of token to not fill the context
        #self.logger.info(self.display_chat_log(messages.append("\nAssistant: "+full_response)))
        return full_response
    

    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append(Message(role=role, content=content))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def reset_session(self):
        """Reset the current session and clear context history."""
        self._initialize_new_session(self.system_message.content)
        self.logger.info("Chat session and context history reset")

    def clear_history(self):
        """Clear conversation and context history."""
        self.conversation_history = []
        self.context_history = []
        self.logger.info("History cleared")

    def _initialize_new_session(self,system_prompt):
        """Initialize a new chat session with fresh context."""
        try:
            ollama.chat(model=self.model_name, messages=[],options={"temperature" : self.temperature})
        except Exception as e:
            self.logger.warning(f"Error clearing Ollama context: {e}")

        self.conversation_history = []
        self.context_history = []  # Initialize context history

        self.system_message = Message(
            role="system",
            content=system_prompt 
        )
        self.logger.info("Initialized new chat session with fresh context")



    '''
    def generate_answer_stream2(self, query: str) -> Generator[str, None, None]:
        """returns full response without streaming"""
        try:
            current_context = None

           
            self.system_message = Message(
                role="system",
                content=self.natural_prompt 
            )


            current_context,filename = self.rag.retrieve_doc_content(query)

            self.add_context_to_history(current_context)
            
            self._add_to_history("user", query)

            messages = self._prepare_messages(True, current_context)

            self.logger.info(self.display_chat_log(messages))
            full_response = ""
            token_count = 0

            for part in ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature" : 0, "num_ctx": 8096}
            ):
                token = part['message']['content']
                full_response += token
                token_count += 1
            
            return full_response,filename
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.logger.error(error_msg)
            return
        
            
    def remove_tag(self,s):
        match = re.match(r'^<([^>]+)>(.*)', s)
        if match:
            return match.group(1), match.group(2)  # Extracted tag and content
        return None, s  # No match, return original string

    def classify_query(self, query: str) -> str:
        """Classifies the query into one of the predefined categories using AI with few-shot examples."""
        classifier_prompt = f"""
        You are an AI classifier. Classify the following question into one of five categories:

        - "syllabus" -> If it is specifically related to the syllabus or subject matter of the course.
        - "teachers" -> If it is related to the teachers or instructors of the course.
        - "schedule" -> If it is related to the class schedule or timing.
        - "exam" -> If it is related to exams or assessments.
        - "materials" -> If it is related to textbooks or study materials.
        - "other" -> if is not related to any other class.

        Here are some examples:

        **Syllabus-Related Questions:**
        1. "What topics are covered in the syllabus for this course?" -> syllabus
        2. "Can you provide an overview of the syllabus?" -> syllabus
        3. "Which topics are treated in the course?" -> syllabus
        4. "What are the main subjects listed in the syllabus?" -> syllabus
        5. "Is [argoument] treated in the course?"  -> syllabus

        **Teacher-Related Questions:**
        6. "Who is teaching the course?" -> teachers
        7. "Who is the professor for this subject?" -> teachers
        8. "Is the course taught by Dr. Smith?" -> teachers

        **Schedule-Related Questions:**
        9. "What is the course schedule?" -> schedule
        10. "What times are the lectures held?" -> schedule
        11. "Where is the class held?" -> schedule

        **Exam-Related Questions:**
        12. "What format will the exam take?" -> exam
        13. "Is the exam oral?" -> exam
        14. "Will there be a project work? " -> exam

        **Textbook/Study Material Questions:**
        15. "Which textbook should I refer to for this course?" -> materials
        16. "Is there a recommended reading list?" -> materials
        17. "Where can I find the course materials?" -> materials

        Now, classify the following question:
        "{query}"

        Respond with ONLY one of the following categories: syllabus, teachers, schedule, exam, materials, other.
        Do not use quatations marks (" ") or quotes (' ')
        """

        classification = ollama.generate(model=self.model_name, prompt=classifier_prompt, stream=False, options={"temperature" : 0} )
        
        return classification.response
        '''