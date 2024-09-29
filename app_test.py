
import streamlit as st

def main():

    st.title("Sentiment Analysis App Test")
    st.subheader("Work in Progress")
    st.text("test")
    
    # Simple slider for testing if Streamlit is functioning
    value = st.slider("Select a value", 0, 100)
    st.write(f"Selected value: {value}")

if __name__ == "__main__":
    main()
