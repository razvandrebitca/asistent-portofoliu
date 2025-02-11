def apply_theme(theme):
    if theme == "Dark":
        return """
        <style>
            body, .stApp {
                background-color: #303030;
                color: white;
            }
            .stButton>button {
                background-color: #565656;
                color: white;
            }
            .stDataFrame {
                background-color: #3e3e3e;
                color: white;
            }
            .stSidebar {
                background-color: #333333;
                color: white;
            }
        </style>
        """
    else:
        return """
        <style>
            body, .stApp {
                background-color: white;
                color: black;
            }
            .stButton>button {
                background-color: #f0f0f0;
                color: black;
            }
            .stDataFrame {
                background-color: white;
                color: black;
            }
            .stSidebar {
                background-color: #f8f9fa;
                color: black;
            }
            h1{
                color: black;
            }
            .st-de {
             color: rgb(0 162 251);
            }
            h3{
                color: black;
            }
            p{
                color: black;
            }
            .st-emotion-cache-1itdyc2 {
                background-color: rgb(232 232 234);
            }
        </style>
        """