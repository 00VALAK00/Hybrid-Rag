from src.architecture.graph_builder import graph 

def main():
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        input_data = {
            "messages":
                [
                    {"role":"user", "content": user_input}
                ]
        }
        for event in graph.stream(input = input_data, 
                                           stream_mode = "values"):
            print(event, end="", flush=True)

if __name__ == "__main__": 
    main()




