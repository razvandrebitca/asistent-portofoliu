import streamlit as st

# Function to ensure session state variables are initialized
def initialize_session_state():
    if 'user' not in st.session_state:
        st.session_state['user'] = "user"
    if 'password' not in st.session_state:
        st.session_state['password'] = "password"
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'registered' not in st.session_state:
        st.session_state['registered'] = False

def register():
    st.title("Register")

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if username in st.session_state['users']:
            st.error("Username already taken! Please choose a different one.")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            st.session_state['users'][username] = password
            st.success("Registration successful! You can now log in.")
            st.session_state['registered'] = True

def login():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state['user'] and st.session_state['password'] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            return 1
        else:
            st.error("Invalid username or password")
            return 0

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.success("You have been logged out.")

def is_logged_in():
    return st.session_state.get('logged_in')
