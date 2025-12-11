from typing import Generator
from g4f.client import Client
from g4f.client import ChatCompletion
import g4f
import g4f.Provider
import tkinter as tk
from tkinter import scrolledtext, END, filedialog
import threading
import g4f.errors
import g4f.providers
import g4f.providers.any_provider # Import g4f errors for more specific handling

class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("g4f Chat GUI")

        self.message_list = [
            {
                "role": "system",
                "content": "You are an expert AI Programming Assistant. Provide clear and accurate code examples, explain complex programming concepts, and help debug code. When providing code, use markdown code blocks.",
            },
        ]
        self.client = Client()

        # Chat display area
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=70, height=20)
        self.chat_area.pack(padx=10, pady=10)

        # Input frame for buttons and entry field
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.upload_button = tk.Button(self.input_frame, text="Upload File", command=self.select_file)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 5))

        self.uploaded_file_label = tk.Label(self.input_frame, text="No file selected.", width=25, anchor="w", justify=tk.LEFT)
        self.uploaded_file_label.pack(side=tk.LEFT, padx=(0,5))

        # Send button (packed to the right)
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(5,0))

        # Input field (expands to fill remaining space)
        self.input_field = tk.Entry(self.input_frame)
        self.input_field.pack(side=tk.LEFT, expand=True, fill=tk.X) # Fills space between label and send button
        self.input_field.bind("<Return>", self.send_message_event)

        # Variables to store uploaded file data
        self.uploaded_files = [] # List to store multiple {name: '...', content: '...'} dicts



    def display_message(self, who: str, content: str, is_first_part: bool = False, is_last_part: bool = False):
        self.chat_area.config(state='normal')
        if is_first_part:
            self.chat_area.insert(tk.END, f"{who}: ")
        self.chat_area.insert(tk.END, content)
        if is_last_part:
             self.chat_area.insert(tk.END, "\n")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def select_file(self):
        filepaths = filedialog.askopenfilenames( # Changed to askopenfilenames
            title="Select files (text, code, etc.)",
            filetypes=(
                ("Text files", "*.txt"),
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("HTML files", "*.html;*.htm"),
                ("CSS files", "*.css"),
                ("Java files", "*.java"),
                ("C/C++ files", "*.c;*.cpp;*.h;*.hpp"),
                ("Markdown files", "*.md"),
                ("JSON files", "*.json"),
                ("XML files", "*.xml"),
                ("All files", "*.*")
            )
        )
        if filepaths:
            self.uploaded_files = [] # Clear previous selection
            loaded_file_names = []
            errors = []

            for filepath in filepaths:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: # Added errors='ignore' for robustness
                        content = f.read()
                    filename = filepath.split('/')[-1]
                    self.uploaded_files.append({"name": filename, "content": content})
                    loaded_file_names.append(filename)
                except Exception as e:
                    errors.append(f"Error loading {filepath.split('/')[-1]}: {e}")
            
            if loaded_file_names:
                if len(loaded_file_names) == 1:
                    display_name = loaded_file_names[0]
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."
                    self.uploaded_file_label.config(text=f"File: {display_name}")
                    self.display_message("System", f"Loaded file: {', '.join(loaded_file_names)}", is_first_part=True, is_last_part=True)
                else:
                    self.uploaded_file_label.config(text=f"{len(loaded_file_names)} files selected")
                    self.display_message("System", f"Loaded {len(loaded_file_names)} files: {', '.join(loaded_file_names)}", is_first_part=True, is_last_part=True)

            if errors:
                for error_msg in errors:
                    self.display_message("System", error_msg, is_first_part=True, is_last_part=True)
                if not loaded_file_names: # If all files failed to load
                    self.uploaded_file_label.config(text="Error loading files.")
        else: # No files selected or dialog cancelled
            self.uploaded_files = []
            self.uploaded_file_label.config(text="No file selected.")

    def send_message_event(self, event=None):
        self.send_message()

    def send_message(self):
        user_typed_message = self.input_field.get()
        
        message_for_ai = user_typed_message
        message_to_display_as_user = user_typed_message

        if self.uploaded_files:
            file_contexts = []
            uploaded_file_names_list = [f["name"] for f in self.uploaded_files]

            for i, uploaded_file in enumerate(self.uploaded_files):
                # Try to guess language for markdown code block from extension
                file_ext = uploaded_file['name'].split('.')[-1].lower()
                lang_hint = file_ext if file_ext in ['py', 'js', 'java', 'c', 'cpp', 'html', 'css', 'md', 'json', 'xml'] else 'text'
                
                file_contexts.append(
                    f"File {i+1}: '{uploaded_file['name']}'\n```{lang_hint}\n{uploaded_file['content']}\n```"
                )
            
            full_file_context_header = "The user has uploaded the following file(s):\n\n" + "\n\n".join(file_contexts) + "\n\n"

            if user_typed_message.strip():
                message_for_ai = f"{full_file_context_header}User's query (potentially related to the file(s)):\n{user_typed_message}"
                print(len(uploaded_file_names_list), "files uploaded, user message:", user_typed_message)
                print("Length of message for AI:", len(message_for_ai))
            else:
                message_for_ai = full_file_context_header
                message_to_display_as_user = f"(Sent content from: {', '.join(uploaded_file_names_list)})"

            # Clear uploaded file info after preparing the message
            self.uploaded_files = []
            self.uploaded_file_label.config(text="No file selected.")
        
        if not message_for_ai.strip(): # If nothing to send (no text, no file used)
            return

        # Display what the user sees
        self.display_message(who="You", content=message_to_display_as_user, is_first_part=True, is_last_part=True)
        
        # Add the potentially augmented message to the history for the AI
        self.message_list.append({"role": "user", "content": message_for_ai})
        self.input_field.delete(0, tk.END)

        # Disable input while processing
        self.input_field.config(state='disabled')
        self.send_button.config(state='disabled')
        self.upload_button.config(state='disabled')

        # Run API call in a separate thread to keep GUI responsive
        threading.Thread(target=self._process_response, daemon=True).start()

    def _process_response(self):
        response_string = ""
        first_chunk = True

        try:
            # user_message = self.message_list[-1]["content"] # This line is not strictly needed here
            response_stream: ChatCompletion = self.client.chat.completions.create(
                model="gpt-4.1", # Current model in your code
                messages=self.message_list,
                provider=g4f.Provider.LegacyLMArena,
                stream=True,
            )

            for i, response_line in enumerate(response_stream):
                if i == 0:
                    print("Using provider:", response_line.provider) # Debugging line to see which provider is used
                # It's good practice to check if response_line is a special object (like FinishReason)
                if isinstance(response_line, g4f.providers.response.FinishReason):
                    break # Or handle as needed
                
                # The error occurs when trying to access .choices, so ensure it exists
                if hasattr(response_line, 'choices') and response_line.choices and \
                   hasattr(response_line.choices[0], 'delta') and response_line.choices[0].delta.content:
                    chunk = response_line.choices[0].delta.content
                    # GUI updates from a thread should be scheduled with root.after
                    self.root.after(0, self.display_message, "Assistant", chunk, first_chunk, False)
                    response_string += chunk
                    if first_chunk:
                        first_chunk = False
            
            if response_string and not first_chunk: # If assistant sent something and it wasn't just the prefix
                self.root.after(0, self.display_message, "Assistant", "", False, True) # Add final newline
            
            if response_string: # Only append if we got a response
                self.message_list.append({"role": "assistant", "content": response_string})

        except RuntimeError as e:
            # This catches the "No response from server after multiple attempts"
            error_msg = (f"Error: {e}\nProvider: LegacyLMArena, Model: gpt-4.1 .\n"
                         "The AI provider might be busy or unavailable. "
                         "Please try again later, or try a different model/provider.")
            self.root.after(0, self.display_message, "System", error_msg, True, True)
        except g4f.errors.ProviderNotFoundError:
            self.root.after(0, self.display_message, "System", "Error: Provider not found. Please check the provider name.", True, True)
        except g4f.errors.ModelNotFoundError:
            self.root.after(0, self.display_message, "System", "Error: Model not found for the selected provider.", True, True)
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"An unexpected error occurred: {type(e).__name__}: {e}"
            self.root.after(0, self.display_message, "System", error_msg, True, True)
            import traceback
            print(traceback.format_exc()) # For developer debugging
        finally:
            # Re-enable input fields regardless of success or failure, using root.after for thread safety
            self.root.after(0, self._enable_inputs)

    def _enable_inputs(self):
        self.input_field.config(state='normal')
        self.send_button.config(state='normal')
        self.upload_button.config(state='normal')
        self.input_field.focus_set() # Put cursor back in input field

def main():
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()