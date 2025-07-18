import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { useState } from 'react';
import Home from './components/Home';
import Footer from './components/Footer';
import Navbar from './components/Navbar';
import MapComponent from './components/MapComponent';
import "./App.css";
import SignInForm from './components/LoginPage';

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  return (
    <div className="App">
      <Router>
      <Navbar setIsSidebarOpen={setIsSidebarOpen} isSidebarOpen={isSidebarOpen}/>
      <div className="">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<SignInForm />} />
            <Route path="/map" element={<MapComponent setIsSidebarOpen={setIsSidebarOpen} isSidebarOpen={isSidebarOpen}/>} />
          </Routes>
      </div>
      </Router>
      <Footer />
    </div>
  )
}

export default App;