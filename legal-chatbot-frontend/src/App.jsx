// App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import './index.css';

// Import your pages and components
import HomePage from './Home';    // Import HomePage component
import SignIn from './SignIn';        // Sign In component
import SignUp from './SignUp';        // Sign Up component

function App() {
  return (
    <Router>
      <Routes>
        {/* Define the home route (landing page) */}
        <Route path="/" element={<HomePage />} />
        
        {/* Define routes for Sign In and Sign Up */}
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
      </Routes>
    </Router>
  );
}

export default App;
