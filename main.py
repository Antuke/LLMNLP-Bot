import argparse
import ollama
import sys
from chatbot import ChatBot
from colorama import init, Fore, Style
from transformers import pipeline
from topic_classifier import GuardRailClassic

# Initialize colorama
init(autoreset=True)

class ChatApp:
    def __init__(self, model_name="qwen3:8b", mode="bert"):
        self.chat_model = model_name
        self.bot = ChatBot(model_name=self.chat_model, max_history=3, temperature=0.1)
        self.rails = True
        self.bertguard = None
        self.guard = None
        self.prev_query = ""
        self.prev_query_score = 0
        self.off_topic_count = 0  # Counter for off-topic queries
        self.decay_factor = 0.9   # Score decay per off-topic query
        self.max_off_topic = 4    # Maximum off-topic queries before resetting context
        self.mode = mode
        self.off_topic_counter = 0
        self.initialize_guardrails()
        self.debug = False

    def initialize_guardrails(self):
        """Loads chosen topic guardrail model"""
        if self.mode == "bert":
            print("Using BERT for input guardrailing")
            self.bertguard = pipeline('text-classification', 
                                    model='./classifiers/fine-tuned-distilbert', 
                                    tokenizer='./classifiers/fine-tuned-distilbert')
        elif self.mode == "logistic":
            print("Using logistic for input guardrailing")
            self.guard = GuardRailClassic("logistic")
        elif self.mode == "svm":
            print("Using SVM for input guardrailing")
            self.guard = GuardRailClassic("svm")
        else:
            raise Exception("Wrong mode selected, possible are [bert, svm, logistic]")

    def get_query_score(self, query, threshold=0.5):
        """Returns True if a query is classified as off-topic, False otherwise. Also return probability of being on-topic"""
        if self.mode == "bert":
            res = self.bertguard(query)
            #(f"res = {res}")
            if res[0]['label'] == 'LABEL_0':
                if (1 - res[0]['score']) < threshold:
                    return True, (1 - res[0]['score'])
                else: 
                    return False, (1 - res[0]['score'])
            return False, res[0]['score']
        if self.mode == "svm" or self.mode == "logistic":
            res = self.guard.classify(query)
            #print(f"res = {res}")
            return (True, res) if res < threshold else (False, res)

        raise Exception("Wrong mode selected, possible are [bert, svm, logistic]")

    def calculate_weighted_score(self, current_score):
        """Calculate weighted score between previous and current query"""
        if not self.prev_query:
            return current_score
        
        # As off_topic_counter increases, previous query weight decreases
        current_prev_weight = max(0, self.prev_weight - (self.off_topic_counter * 0.2))
        current_weight = 1 - current_prev_weight
        
        weighted_score = (self.prev_query_score * current_prev_weight + 
                         current_score * current_weight)
        
        if self.debug:
            print(f"DEBUG: Weighted calculation:")
            print(f"Previous score: {self.prev_query_score} with weight {current_prev_weight}")
            print(f"Current score: {current_score} with weight {current_weight}")
            print(f"Final weighted score: {weighted_score}")
        
        return weighted_score
    
    def _handle_offtopic(self):
        self.prev_query = ""
        self.prev_query_score = 0
        self.off_topic_counter = 0
        self.prev_weight = 0.9
            
        self.print_message("Sorry, but it appears that your question is outside my scope.\n"
                        "If you think it's a mistake, try to rephrase it, or disable guardrailing by typing 'rails-off' without quotations.\n", 
                                Fore.RED)

    def check_guardrails(self, query, threshold=0.4):
        # Get current query score
        _,current_score = self.get_query_score(query)
        

        if current_score < threshold:
            # current query is evaluated as off-topic
            self.off_topic_counter += 1
            
            # If too many off-topic queries, reset context
            if self.off_topic_counter > 3:
                self._handle_offtopic()
                return False

            weighted_score = self.calculate_weighted_score(current_score)
            # check if current_score weighted with previous score pass the threshold
            if weighted_score < threshold:
                self._handle_offtopic()
                return False
            
            return True # passed with weighted_score
        # Query passed weighted threshold check
        
        if current_score > self.prev_query_score:
            self.prev_query = query
            self.prev_query_score = current_score
            self.off_topic_counter = 0  # Reset counter for on-topic query
            self.prev_weight = 0.9  # Reset weight
            #print(f"DEBUG: Updated context with new weighted score: {current_score}")
            
        return True


    def print_message(self, message, color=Fore.CYAN):
        """styled terminal print messages"""
        print(f"{color}{message}{Style.RESET_ALL}", end="", flush=True)

    def handle_query(self, query):
        """checks for system command, topic-guardrail and calls chatbot"""
        if query.lower() == "rails-off":
            self.rails = False
            print("Rails are off! To re-enable them type 'rails-on'")
            return
        if query.lower() == "rails-on":
            self.rails = True
            self.off_topic_count = 0  # Reset counter when rails are turned back on
            return
        if query.lower() == "exit":
            self.bot.reset_session()
            self.print_message("Exiting...", Fore.RED)
            exit()
        if query.lower() == "reset":
            self.bot.reset_session()
            self.off_topic_count = 0
            self.prev_query = ""
            self.prev_query_score = 0
            self.print_message("Session reset.", Fore.YELLOW)
            return
        if query == "":
            return

        if self.rails and not self.check_guardrails(query):
            return

        self.print_message("Assistant: ", Fore.BLUE)
        for token in self.bot.answer_rag(query):
            self.print_message(token, Fore.CYAN)
        print("\n")

    def run(self):
        self.print_message("Hello, I'm the AI-assistant for the University of Salerno NLP and LLM course!\n")
        while True:
            query = input(f"\n{Fore.GREEN}Query: {Style.RESET_ALL}").strip()
            self.handle_query(query)

def main():
    parser = argparse.ArgumentParser(description="NLP Course Chatbot")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Ollama model to use for the chatbot (default: qwen3:8b)")
    parser.add_argument("--mode", type=str, default="bert", choices=["bert", "svm", "logistic"], help="Topic classifier mode (default: bert)")
    args = parser.parse_args()
    print("\n\n")
    # Check if the model exists in Ollama
    try:
        models_info = ollama.list()
        
        # Extract model names. ollama.list() returns a dict with 'models' key, which is a list of dicts.
        # Each dict has a 'name' key.
        available_models = [m.model for m in models_info.models]
        

        model_exists = args.model in available_models

        if not model_exists:
            print(f'{args.model} has no exact match in the available models: {available_models}')
            print(f'Proceeding without verification...')

            model_prefixed = args.model + ":latest"
            model_exists = model_prefixed in available_models
            if model_exists:
                print(f'{model_prefixed} found in the available models: {available_models}')
                args.model = model_prefixed
            else:
                print(f'{Fore.RED}{model_prefixed} has no exact match in the available models: {available_models}{Style.RESET_ALL}')
                print(f'{Fore.YELLOW}Proceeding without verification...application may crash if the model is not present in the Ollama library. Be sure to run "ollama pull {args.model}" to download the model if present in the Ollama library.{Style.RESET_ALL}')

    except Exception as e:
        print(f"{Fore.RED}Warning: Could not verify Ollama models: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Proceeding without verification...{Style.RESET_ALL}")

    try:
        app = ChatApp(model_name=args.model, mode=args.mode)
        app.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()

