# **Agentic RAG Chatbot Overview**

The **Agentic RAG Chatbot** is a hybrid AI assistant that integrates **Retrieval-Augmented Generation (RAG)** with knowledge base search and live web search.  
This combination enables the chatbot to deliver accurate, context-aware responses across various domains.  

**GitHub:** [https://github.com/PhiVanDat123/Agentic-RAG-chatbot](https://github.com/PhiVanDat123/Agentic-RAG-chatbot)

---

## **Features**

- **Hybrid Search Capabilities:** Combines internal knowledge base search with live web search to provide comprehensive answers.  
- **Context-Aware Responses:** Utilizes RAG to generate responses that consider the context of the query.  
- **Multi-Source Information Retrieval:** Fetches and processes information from multiple sources to enhance response accuracy.  

---

## **Architecture**

The system is designed with a **modular architecture** to ensure scalability and maintainability.  
Key components include:

- **Backend:** Handles the core logic, including RAG processing and integration with search APIs.  
- **Frontend:** Provides a user interface for interacting with the chatbot.  
- **Configuration:** Manages settings for search APIs, RAG parameters, and other system configurations.  

## **Installation**

### **Clone the Repository**
```bash
git clone https://github.com/PhiVanDat123/Agentic-RAG-chatbot.git
cd Agentic-RAG-chatbot
```

## **Install Dependencies**

Ensure you have **Python 3.8+** installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

## **Set Up Environment Variables**

Create a .env file and configure the necessary environment variables, such as **API keys** for search services.

## **Run the Application**

Start the backend and frontend services:

python run_backend.py
python frontend.py

## **Usage**

Once the application is running, navigate to http://localhost:5000
 in your web browser.
Enter your query in the chat interface, and the chatbot will process it using RAG and provide a response.

## **Contributing**

Contributions are welcome! To contribute:

Fork the repository.

Create a new branch for your feature or fix.

Commit your changes with clear messages.

Push to your fork and create a pull request.

## **License**

This project is licensed under the **MIT License**. See the LICENSE file for details.

For more detailed information and updates, please refer to the **GitHub repository**.
