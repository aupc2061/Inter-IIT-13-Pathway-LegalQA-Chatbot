import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import the useNavigate hook
import './SignIn.css';

const SignIn = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState(""); // For displaying error messages
  const navigate = useNavigate(); // Create an instance of the navigate function

  // Handle the form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Reset error message
    setErrorMessage("");

    try {
      // Send a POST request to the sign-in endpoint
      const response = await fetch("http://localhost:5000/api/signin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // Include cookies in the request
        body: JSON.stringify({ username: email, password }),
      });

      // Handle response
      if (response.ok) {
        const data = await response.json();
        console.log("Sign-in successful:", data.message);
        navigate("/"); // Redirect to home page on successful sign-in
      } else {
        const data = await response.json();
        setErrorMessage(data.message || "Something went wrong!");
      }
    } catch (error) {
      setErrorMessage("Network error: " + error.message);
      console.error("Error signing in:", error);
    }
  };

  return (
    <section className="sign-in-section">
      <div className="container">
        <div className="sign-in-form">
          <h2>Sign In</h2>
          <p>Enter your details to sign in!</p>

          {errorMessage && <p className="error-message">{errorMessage}</p>}

          <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email:</label>
            <InputBox
              type="email"
              name="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <InputBox
              type="password"
              name="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button type="submit" className="submit-button">
            Sign In
          </button>
          </form>

          <p>
            Don’t have an account?{" "}
            <a href="/signup">Sign Up</a>
          </p>
        </div>
      </div>
    </section>
  );
};

const InputBox = ({ type, placeholder, name, value, onChange }) => {
  return (
    <div className="input-box">
      <input
        type={type}
        placeholder={placeholder}
        name={name}
        value={value}
        onChange={onChange}
      />
    </div>
  );
};

export default SignIn;
