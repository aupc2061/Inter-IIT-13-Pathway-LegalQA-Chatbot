import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './SignUp.css';

function SignUp() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState(''); // Add state for name
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Make sure all fields are filled
    if (!username || !password || !name) {
      setErrorMessage('Please fill in all fields.');
      return;
    }

    // Send sign-up data to the backend
    try {
      const response = await fetch('http://localhost:5000/api/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password, name }),
      });

      const data = await response.json();
      if (response.ok) {
        navigate('/signin'); // Redirect to sign-in page after successful sign-up
      } else {
        setErrorMessage(data.message); // Show error message if something goes wrong
      }
    } catch (error) {
      console.error('Error:', error);
      setErrorMessage('An error occurred. Please try again.');
    }
  };

  return (
<div className="sign-up">
  <div className="sign-up-form">
    <h2>Create an Account</h2>
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="name">Full Name:</label>
        <input
          type="text"
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter your full name"
        />
      </div>

      <div className="form-group">
        <label htmlFor="username">Email:</label>
        <input
          type="text"
          id="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter an email"
        />
      </div>

      <div className="form-group">
        <label htmlFor="password">Password:</label>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter a password"
        />
      </div>

      {errorMessage && <p className="error-message">{errorMessage}</p>}

      <button type="submit" className="submit-button">
        Sign Up
      </button>

      <p> Already have an account?<a href="/signin"> Sign In</a> </p>
      
    </form>
  </div>
</div>

  );
}

export default SignUp;
