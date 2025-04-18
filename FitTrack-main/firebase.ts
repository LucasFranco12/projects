import { initializeApp } from 'firebase/app';
import { initializeAuth } from 'firebase/auth';
import * as firebaseAuth from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';
// Note: getAnalytics doesn't work directly with Expo unless you use EAS or custom native code
// import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCXPib7yEs9gkuP5vymaRWYDMZ1EuDaWaE",
  authDomain: "fittrack-efb1a.firebaseapp.com",
  projectId: "fittrack-efb1a",
  storageBucket: "fittrack-efb1a.firebasestorage.app",
  messagingSenderId: "852169786355",
  appId: "1:852169786355:web:13298d6228b13b8d084886",
  measurementId: "G-K87E4RM4E8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// Analytics doesn't work directly in Expo managed workflow
// const analytics = getAnalytics(app);

// Initialize Auth with AsyncStorage persistence
const auth = initializeAuth(app, {
  persistence: (firebaseAuth as any).getReactNativePersistence(AsyncStorage)
});

const db = getFirestore(app);

export { auth, db }; 