import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Home.css';
import { Drawer, Button, List, ListItem, ListItemText, Divider } from '@mui/material';
import BorderColorIcon from '@mui/icons-material/BorderColor';
import SendIcon from '@mui/icons-material/Send';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';


function Home() {
  const [userMessage, setUserMessage] = useState('');
  const [messages, setMessages] = useState([]); // Store chat messages (user and bot)
  const [isDarkMode, setIsDarkMode] = useState(false); // Dark mode toggle
  const [thoughtProcess, setThoughtProcess] = useState([]); // Track thought process
  const [user, setUser] = useState(null); // Store user object (including name)
  const [isMessageSent, setIsMessageSent] = useState(false); // Track message sent state
  const [typeOptions, setTypeOptions] = useState([]); // Store dynamic types
  const [expandedTypes, setExpandedTypes] = useState({}); // Track which message types are expanded
  const [chatID, setChatID] = useState(0);
  const [userChats, setUserChats] = useState([]); // Store user chats
  const [selectedChat, setSelectedChat] = useState(null); // Store selected chat
  const [feedback, setFeedback] = useState(''); // Store human feedback
  const [showInput, setShowInput] = useState(true); // Show input field for user message
  const [showPopup, setShowPopup] = useState(false); // Show popup for feedback

  const navigate = useNavigate();

  const typeMap = {
    plan_and_schedule: 'Plan & Schedule...',    // Planning and scheduling tasks
    join: 'Decision Making...',                 // Making decisions like replan or finalize
    rewrite: 'Rewriting & Refining...',         // Refining or rewriting responses
    generate: 'Generate Response...',           // Final answer generation
    human_input_node: "Human's feedback..."    // Human feedback node
  };

  useEffect(() => {
    const fetchThoughtProcess = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/get-thought-process?' + new Date().getTime());
        const data = await response.json();
       // console.log("data from fetch thought process: ", data);
        if (Array.isArray(data.thoughtProcess)) {
         // console.log("thoughtprocess is an array");
          setThoughtProcess(data.thoughtProcess);  // Set it to the fetched data
        } else {
          console.error('Expected thoughtProcess to be an array');
        }
      } catch (error) {
        console.error('Error fetching thought process:', error);
      }
    };
  
    const interval = setInterval(fetchThoughtProcess, 100);  // Poll every 5 seconds
  
    return () => clearInterval(interval);  // Cleanup on unmount
  }, []);
  

  useEffect(() => {
    //console.log('Fetched Thought Process:', thoughtProcess);
  }, [thoughtProcess]);
  
  

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:5001'); // WebSocket connection to backend
  
    ws.onopen = () => {
      console.log('Connected to WebSocket server');
    };
  
    ws.onmessage = (message) => {
      if (message.data === 'triggerPopup') {
        setShowPopup(true); // Show feedback popup
      }
    };
  
    return () => {
      ws.close(); // Cleanup WebSocket on component unmount
    };
  }, []);
  
  // Function to handle user feedback change
  const handleFeedbackChange = (e) => {
    setFeedback(e.target.value);
  };
  // Function to handle feedback submission

  const emptyTextFiles = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/empty-text-files', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
    }
    catch (error) {
      console.error('Error emptying text files:', error);
    }
  };

  const [open, setOpen] = useState(true);
  const [secondOpen, setSecondOpen] = useState(false);
  const toggleSidebar = (num) => {
    if(num===1){
      setOpen(!open);
    }
    else{

    }
  };
  const toggleSecondSidebar = () => {
    setSecondOpen(!secondOpen);
  };
  const handleFeedbackSubmit = async () => {
    if (!feedback.trim()) return;

    try {
      const response = await fetch('http://localhost:5000/api/submit-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ feedback }),  // Send feedback to the server
      });

      if (response.ok) {
        //alert('Feedback submitted successfully!');
        setFeedback(''); // Clear feedback after submission
        setShowPopup(false); // Close popup after feedback submission
      } else {
        alert('Failed to submit feedback. Please try again later.');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  useEffect(() => {
    const fetchUserChats = async () => {
     // console.log("entered user chats fetch")
      try {
        const response = await fetch(`http://localhost:5000/api/getUserChats?chatID=${chatID}`, {
          method: 'GET',
          credentials: 'include',
        });

        const data = await response.json();
       // console.log("data from fetch user chats: ", data)
        if (data.chats) {
          setUserChats(data.chats); // Assuming the response has 'chats' key
          //setChatID(data.chats.length); // Set chatID based on the number of chats
        //  console.log('User chats:', userChats);
        }
      } catch (error) {
        console.error('Error fetching user chats:', error);
      }
    };

    fetchUserChats();
  }, []); // Empty dependency array ensures it runs only once after the first render

  const getChatNumber = async () => {
  //  console.log("entered user chats fetch")
    try {
      const response = await fetch(`http://localhost:5000/api/getUserChats?chatID=${chatID}`, {
        method: 'GET',
        credentials: 'include',
      });

      const data = await response.json();
    //  console.log("data from fetch user chats: ", data)
      if (data.chats) {
        setUserChats(data.chats); // Assuming the response has 'chats' key
        //setChatID(data.chats.length); // Set chatID based on the number of chats
       // console.log('User chats:', userChats);
      }
    } catch (error) {
      console.error('Error fetching user chats:', error);
    }
  };


  const createNewChat = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/new-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include credentials to pass the session
        body: JSON.stringify({ username: user.username }), // Assuming the username is available
      });
      const data = await response.json();
      setChatID(data.chat_num); // Store the chat ID in the state
      setMessages([]); // Clear the chat messages
      setUserMessage(''); // Clear the input field
      setSelectedChat(null); // Clear the selected chat
      setShowInput(true); // Show the input field for user message
      emptyTextFiles(); // Empty text files before creating a new chat
      console.log('New chat created, chat ID:', data.chat_num);
    } catch (error) {
      console.error('Error creating new chat:', error);
    }
  };

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));


  const fetchAnsThought = async () => {
    //console.log("entered fetchAnsThought")
    //console.log("chatID in fetchAnsThought: ", chatID)
   // console.log("user in fetchAnsThought: ", user)
    if (user) {
      try {
        const response = await fetch(`http://localhost:5000/api/getChats?chatID=${chatID}`, {
          method: 'GET',
          credentials: 'include', // Include credentials to pass the session/cookie
        });

        const data = await response.json();
       // console.log("data from fetch ans thought: ", data)

        if (data.chats) {
          const allMessages = [];
          const encounteredTypes = new Set(); // Track encountered types dynamically
         // console.log("print data chats in get chats: ", data.chats) //i have

          // Iterate through each chat record
          data.chats.forEach(chat => {
            const new_chat = chat.scriptResponse;
            const finalAnswer = new_chat.final_ans; // Only grab the final answer, not intermediate responses
           // console.log("final answer from chat: ", finalAnswer)
            // Capture the user's query and add it to the chat history
            const userMessage = chat.usermessage; // Assuming the user message is stored as user_message
           // console.log("user message from chat: ", userMessage)
            if (userMessage) {
              allMessages.push({
                sender: 'user',
                text: userMessage,
              });
            }

            // Add the final answer from the bot to the chat history
            if (finalAnswer) {
              allMessages.push({
                sender: 'bot',
                text: finalAnswer,
              });
            }

            setMessages(allMessages);

            // Handle types (Plan, Join, etc.) for the thought process display
            const messages = new_chat.state_messages || [];
            messages.forEach(messageObj => {
              Object.entries(messageObj).forEach(([type, messageArray]) => {
                messageArray.forEach((message) => {
                  const processedMessage = message.includes('ERROR')
                    ? 'Encountered error, thinking again'
                    : message;

                  allMessages.push({
                    type: typeMap[type] || type,  // Use the mapped type or fallback to the original if not found
                    text: processedMessage,
                  });

                  // Add type to the set of encountered types
                  encounteredTypes.add(typeMap[type] || type);
                });
              });
            });
          });
         // console.log("all messages in fetchAnsThought: ", allMessages)

          // Convert the encountered types set to an array
          setTypeOptions([ ...Array.from(encounteredTypes)]);

          // Set the flattened messages as the chat history
          
          setMessages(allMessages);
        
          //console.log("all messages in set messages: ", messages)
         // setThoughtProcess(allMessages)
        }
      } catch (error) {
        console.error('Error fetching chat history:', error);
      }
    }
  };

  const toggleTypeVisibility = (type) => {
    setExpandedTypes(prev => ({
      ...prev,
      [type]: !prev[type], // Toggle visibility for the selected type
    }));

    getChatNumber();

  };


  const sendMessage = async () => {
    if (!userMessage.trim()) return;
    
    // Add the user's query to the chatbox
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'user', text: userMessage },
    ]);
    //console.log("set user message: ", messages)
    //setThoughtProcess('Processing your request...');

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ message: userMessage, chatID: chatID }),  // Include chatID in the request body
      });
      console.log("chat API completed");

      const data = await response.json();
      const finalAnswerResponse = data.response.final_ans; // Assuming the response has a 'final_ans'
     
      // Add the bot's response to the messages
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'bot', text: finalAnswerResponse },
      ]);

      //console.log("messages after final response added: ", messages)

      // After chat is complete, update state to trigger the fetch for chat history
      setIsMessageSent(true);
      setUserMessage(''); // Clear the input field after sending
      

      // Trigger fetchAnsThought after message is sent
      fetchAnsThought();
    } catch (error) {
      console.error('Error sending message:', error);
     // setThoughtProcess('Error processing your request.');
    }
  };

  const handleSignOut = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/signout', { 
        method: 'POST', 
        credentials: 'include' 
      });

      if (response.ok) {
        console.log('User signed out successfully.');
        setUser(null); // Reset user state
        navigate('/'); // Redirect to home or login page
        alert('Sign out successful!');
      } else {
        console.error('Sign out failed');
      }
    } catch (error) {
      console.error('Error signing out:', error);
      alert('Error signing out. Please try again later.');
    }
  };

  const handleChatClick = (chatNum) => {
    // Filter all messages with the selected chat number
    const chatHistory = userChats.filter(chat => chat.chat_num === chatNum); // Get all chats with the selected chat_num
    setChatID(chatNum); // Set the chat ID to the selected chat number
    setShowInput(false); // Show the input field for user message
    emptyTextFiles(); // Empty text files before fetching chat history
    
    // Create an array to store selected messages
    const selectedChatMessages = [];
  
    // Loop through each chat history and extract user message and final answer
    chatHistory.forEach(chat => {
      const userMessage = chat.usermessage;  // Assuming 'usermessage' is the user's input
      const finalAnswer = chat.final_ans;  // Assuming 'final_ans' is the bot's response
      
      // Add user message if available
      if (userMessage) {
        selectedChatMessages.push({ sender: 'user', text: userMessage });
      }
      
      // Add final answer if available
      if (finalAnswer) {
        selectedChatMessages.push({ sender: 'bot', text: finalAnswer });
      }
    });
  
    // Set the selected chat messages to state
    setSelectedChat(selectedChatMessages);
    setMessages([]);
    setMessages(selectedChatMessages);
    console.log("Filtered chat history on click:", selectedChatMessages);
  };
  


  // Fetch the current signed-in user from the session when the component mounts
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/current-user', { method: 'GET', credentials: 'include' });
        const data = await response.json();
        if (data.user) {
          setUser(data.user); // Set the user object (including name) from the session
        }
      } catch (error) {
        console.error('Error fetching user:', error);
      }
    };

    fetchUser();
  }, []);

  // ... other existing code ...

  return (
    <div className={`App ${isDarkMode ? 'dark' : 'light'}`}>
      {user && (
        <div className="user-greeting">
          <span>Hello, {user.name}</span>
        </div>
      )}

      <div className="auth-buttons">
        {user ? (
          <div className="user-info">
            <button className="auth-button" onClick={handleSignOut}>
              Sign Out
            </button>
          </div>
        ) : (
          <>
            <Link to="/signin">
              <button className="auth-button">Sign In</button>
            </Link>
            <Link to="/signup">
              <button className="auth-button">Sign Up</button>
            </Link>
          </>
        )}
      </div>

      <div className="main-content">
        {user && chatID == 0 && (
          <div className="welcome-message">
            <p className='welcome-p'>Welcome to your legal assistant! </p>
            <button onClick={createNewChat} className='create_new_chat_btn'>
              Create New Chat
            </button>
          </div>
        )}

        <div>
          <ArrowForwardIosIcon onClick={() => toggleSidebar(1)} style={{ 
            cursor: 'pointer',
            position: 'fixed', // Keep it on the screen at all times
            left: '10px',     // Position it on the right edge
            top: '50%',        // Vertically center it (optional)
            transform: 'translateY(-50%)', // Adjust for centering
            zIndex: 1000       // Ensure it is above other elements
          }} />

          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
            <ArrowBackIosIcon onClick={toggleSecondSidebar} style={{ 
              cursor: 'pointer',
              position: 'fixed', // Keep it on the screen at all times
              right: '10px',     // Position it on the right edge
              top: '50%',        // Vertically center it (optional)
              transform: 'translateY(-50%)', // Adjust for centering
              zIndex: 1000       // Ensure it is above other elements
              }} />
          </div> 
          
          <Drawer
            anchor="left"
            open={open}
            onClose={() => toggleSidebar(1)}
            variant="persistent"
            // onClose={toggleDrawer(false)} // Drawer will not close on clicking the backdrop due to onClose customization
            ModalProps={{
              BackdropProps: {
                invisible: true, // Make the backdrop transparent
              },
            }}
            sx={{
              width: 250,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: 250,
                boxSizing: 'border-box',
              },
            }}
            style={{fontFamily: "sans-serif"}}
          >
    
            <div className="bot-name">
              <h1>PathLex</h1>
              <ArrowBackIosIcon onClick={() => toggleSidebar(1)} style={{ 
                cursor: 'pointer',
                position: 'fixed', // Keep it on the screen at all times
                left: '210px',     // Position it on the right edge
                top: '1.5%',        // Vertically center it (optional)
                zIndex: 1000       // Ensure it is above other elements
              }} />
            </div>

            <Divider />

            <div className="chatbuttons" style={{ padding: '20px' }}>
              {userChats.length > 0 ? (
                // Sort chats in descending order and map over them
                [...new Set(userChats.map(chat => chat.chat_num))] // Remove duplicates
                  .sort((a, b) => b - a) // Sort in descending order
                  .map((chatNum, index) => (
                    <Button
                      key={index}
                      variant="outlined"
                      fullWidth
                      onClick={() => handleChatClick(chatNum)}
                      sx={{
                        marginBottom: '10px',
                        padding: '10px',
                        fontSize: '14px', // Adjust font size if needed
                      }}
                    >
                      Chat {[...new Set(userChats.map(chat => chat.chat_num))].length - index  } {/* Display "Chat 1", "Chat 2", "Chat 3", etc. */}
                    </Button>
                  ))
              ) : (
                <p>No chats available.</p>
              )}
            </div>
            <div className="top-left-button" style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
              <button onClick= {createNewChat} className='start_chat_btn'>Create New Chat</button>
            </div>
          </Drawer>
        </div>


        <div className="chat-container">
          <div className="chatbox">
            <div className="messages">
              {(selectedChat || messages).filter(msg => msg.sender === "user" || msg.sender === "bot").map((msg, idx) => (
                <div key={idx} className={`message ${msg.sender}`}>
                  <span>{msg.text}</span>
                </div>
              ))}
            </div>

            <div className="feedback-box">
              {showPopup && (
                <div className="popup">
                  <h3>Provide Feedback</h3>
                  <input
                    type = "text"
                    value={feedback}
                    onChange={handleFeedbackChange}
                    placeholder="Enter your feedback..."
                  />
                  <SendIcon onClick={handleFeedbackSubmit} style={{ cursor: 'pointer', marginTop: "10px", color: "blue" }} />               
                </div>
              )}
            </div>

            {showInput && chatID!=0 && (
              <div className="input-area" style={{backgroundColor: "white", paddingRight: "20px", borderRadius: "8px"}}>
                <input
                  type="text"
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  placeholder="Ask a legal question..."
                  style={{border: "none"}}
                />
                <img src='/send_btn.svg' style={{cursor: "pointer"}} onClick={sendMessage}></img>
              </div>
            )}
          </div> {/*chatbox*/}
          
          <Drawer
            anchor="right"
            open={secondOpen}
            onClose={toggleSecondSidebar}
            sx={{
              width: 500,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: 400,
                boxSizing: 'border-box',
              },
            }}
          >
<div className="thought-process">
  <div className='thought-header'>
    <ArrowForwardIosIcon onClick={toggleSecondSidebar} style={{ 
      cursor: 'pointer',
      marginTop: '3px',
      zIndex: 1000
    }} />
    <h2>Thought Process</h2>
  </div>
  <div className="type-dropdowns">
    {thoughtProcess.map((item, index) => (
      <div key={index} className="dropdown-container">
        <div className="dropdown-header" onClick={() => toggleTypeVisibility(item.type)}>
          <span>{typeMap[item.type]}</span>
          <span className={`arrow ${expandedTypes[item.type] ? 'up' : 'down'}`}>⯆</span>
        </div>
        {expandedTypes[item.type] && (
          <div className="dropdown-content">
            <div 
              className="message" 
              style={{
                color: item.type === '' ? 'purple' : 'inherit',
                whiteSpace: "pre-line"
              }}
            >
              {/* Highlight specific strings in red and purple */}
              <span>
                {item.text.split(/("source"|"page_number"|"page_content"|"Thought:|"Context from last attempt:)/).map((part, idx) => {
                  if (part === '"source"' || part === '"page_number"' || part === '"page_content"') {
                    return (
                      <span key={idx} style={{ color: 'red' }}>{part}</span>
                    );
                  } else if (part === '"Thought:' || part === '"Context from last attempt:') {
                    return (
                      <span key={idx} style={{ color: 'purple' }}>{part}</span>
                    );
                  } else {
                    return <span key={idx}>{part}</span>;
                  }
                })}
              </span>
            </div>
          </div> // Dropdown content
        )}
      </div> // Dropdown container
    ))}
  </div> {/*Type dropdowns*/}
</div> {/*thought process*/}



          </Drawer>
        </div> {/*chat-container*/}

      </div> {/*main-content*/}
    </div> // App
  );
}


export default Home;
