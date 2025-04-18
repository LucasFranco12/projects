import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { NavigationContainer } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList, MainTabParamList } from './types';
import { COLORS } from '../constants/theme';

import HomeScreen from '../screens/HomeScreen';
import WorkoutsScreen from '../screens/WorkoutsScreen';
import NutritionScreen from '../screens/NutritionScreen';
import ProfileScreen from '../screens/ProfileScreen';
import FoodSearchScreen from '../screens/FoodSearchScreen';
import CreateCustomMealScreen from '../screens/CreateCustomMealScreen';
import OnboardingScreen, { OnboardingData } from '../components/OnboardingScreen';
import { UserData } from '../App';

const Tab = createBottomTabNavigator<MainTabParamList>();
const Stack = createNativeStackNavigator<RootStackParamList>();

interface AppNavigatorProps {
  userData: UserData | null;
  onLogout: () => void;
  showOnboarding?: boolean;
  onOnboardingComplete?: (data: OnboardingData) => void;
  onUpdateUserData: (userData: UserData) => void;
}

interface MainTabsProps {
  userData: UserData;
  onLogout: () => void;
  onUpdateUserData: (userData: UserData) => void;
}

const MainTabs = ({ userData, onLogout, onUpdateUserData }: MainTabsProps) => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.subtext,
        tabBarStyle: {
          paddingBottom: 5,
          paddingTop: 5,
        },
        headerShown: false,
      }}
    >
      <Tab.Screen 
        name="HomeTab" 
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home" color={color} size={size} />
          ),
          title: 'Home',
        }}
      >
        {() => <HomeScreen userData={userData} />}
      </Tab.Screen>
      
      <Tab.Screen 
        name="WorkoutsTab" 
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="barbell" color={color} size={size} />
          ),
          title: 'Workouts',
        }}
      >
        {() => <WorkoutsScreen userData={userData} />}
      </Tab.Screen>
      
      <Tab.Screen 
        name="NutritionTab" 
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="restaurant" color={color} size={size} />
          ),
          title: 'Nutrition',
        }}
      >
        {() => <NutritionScreen userData={userData} onUpdateUserData={onUpdateUserData} />}
      </Tab.Screen>
      
      <Tab.Screen 
        name="ProfileTab" 
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person" color={color} size={size} />
          ),
          title: 'Profile',
        }}
      >
        {() => (
          <ProfileScreen 
            userData={userData} 
            onLogout={onLogout} 
            onUpdateUserData={onUpdateUserData}
          />
        )}
      </Tab.Screen>
    </Tab.Navigator>
  );
};

const AppNavigator = ({ 
  userData, 
  onLogout, 
  showOnboarding, 
  onOnboardingComplete,
  onUpdateUserData 
}: AppNavigatorProps) => {
  // If onboarding is required, show the onboarding screen
  if (showOnboarding && onOnboardingComplete) {
    return (
      <NavigationContainer>
        <OnboardingScreen onComplete={onOnboardingComplete} />
      </NavigationContainer>
    );
  }

  // If userData is null, don't render anything
  if (!userData) return null;

  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="MainTabs">
          {() => <MainTabs userData={userData} onLogout={onLogout} onUpdateUserData={onUpdateUserData} />}
        </Stack.Screen>
        <Stack.Screen name="FoodSearch" component={FoodSearchScreen} />
        <Stack.Screen 
          name="CreateCustomMeal" 
          component={CreateCustomMealScreen as any}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator; 