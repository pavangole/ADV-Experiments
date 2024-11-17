import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sklearn

if 'chats' not in st.session_state:
    st.session_state.chats = {"Chat 1": []}  # Initialize with a default chat
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if 'first_run' not in st.session_state:
    st.session_state.first_run = {}

# Function to prepend the prompt format
def prepend_prompt_format(prompt, data):
    return f"Your task is to give answer in two sections, First section will begin with #ANSWER# and following it would be one to two line answer. Second section will be code (if applicable) and it will begin with #CODE# followed by python code which will always be related to matplotlib visualization. Also assume that data is present in the 'data' variable. Don't modify data variable directly. Also remember that the code you generate will be given in the exec() function of python. Don't mention anything about code the user should not know that there is code. If code is not required at all or no visualization is asked, then create empty section #CODE#nocode. Generate Python code to visualize this dataset: {data.head().to_string()}.\nQuery: {prompt}"

# Display user messages
def show_user_message(message):
    st.chat_message("user").write(message['parts'][0])

# Execute chart code
def exec_chart_code(code, data):
    if not code:
        return None
    try:
        exec_locals = {}
        exec(code, {"plt": plt, "sns": sns, "pd": pd, 'sklearn': sklearn, "data": data}, exec_locals)
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        return plot_buffer
    except Exception as e:
        st.error(f"Error in executing the generated code: {str(e)}")
        return None

# Display assistant messages
def show_assistant_message(message):
    answer = message.parts[0].text.split("#ANSWER#")[1].split("#CODE#")[0].strip()
    code = message.parts[0].text.split("#CODE#")[1].strip()
    if code.startswith('```python'):
        code = code[9:-3]
    if code.startswith('nocode'):
        code = ''
    st.chat_message("assistant").write(answer)
    if code:
        plot_buffer = exec_chart_code(code, data)
        if plot_buffer:
            st.image(plot_buffer)









# Configure generative AI
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

st.set_page_config(page_title="Visualize GPT", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    .main { background-color: #f4f4f4; padding: 1rem; }
    .stFileUploader { border: 2px dashed #6c63ff; padding: 1rem; border-radius: 10px; background: #fff; }
    .chat-container { background: #ffffff; border: 1px solid #ccc; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Visualize GPT")

st.sidebar.title("Chat Management")

# Sidebar for chat managemenor on last instruction day of semestert
chat_names = list(st.session_state.chats.keys())
selected_chat = st.sidebar.selectbox("Select Chat", chat_names + ["+ New Chat"])

if selected_chat == "+ New Chat":
    new_chat_name = st.sidebar.text_input("Enter a name for the new chat:")
    if st.sidebar.button("Create Chat"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            st.rerun()  # Updated to the correct method

else:
    st.session_state.current_chat = selected_chat

# Set current chat messages
messages = st.session_state.chats[st.session_state.current_chat]
if st.session_state.current_chat not in st.session_state.first_run:
    st.session_state.first_run[st.session_state.current_chat] = False

st.markdown("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    print(data.head().to_string())
    st.write("Here's a preview of your dataset:", data.head())

    st.write("### Chat Section")

    
    # history
    for message in messages:
        if hasattr(message, 'role') and message.role == 'model':
            show_assistant_message(message)
        elif message['role'] == 'user':
            show_user_message(message)

    if not st.session_state.first_run:
        st.session_state.first_run = True
        first_message = [
            {
                'role': 'user',
                'parts': [prepend_prompt_format("Tell me few lines about the dataset and then Show 4 visualization in a single plot that will be the most relevant for this dataset (let them all be of different chart type like scatterplot, histogram, boxplot, etc)", data)]
             }]
        
        response = model.generate_content(
            first_message
        )
        messages.append(response.candidates[0].content)    
        show_assistant_message(messages[-1])
        

    prompt = st.chat_input("Give the prompt")

  
    if prompt:

        messages.append(
            {
                'role': "user",
                'parts': [prompt]
            }
        )

        show_user_message(messages[-1])

        conversation = []

        for message in messages:
            if hasattr(message, 'role') and message.role == 'model':
                conversation.append(message)
            elif message['role'] == 'user':
                conversation.append({
                    'role': 'user',
                    'parts': [prepend_prompt_format(message['parts'][0], data)]
                })


        response = model.generate_content(
            conversation
        )

        
        messages.append(response.candidates[0].content)

        show_assistant_message(messages[-1])
