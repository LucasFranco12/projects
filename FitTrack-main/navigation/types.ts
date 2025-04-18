import { USDAFoodItem } from '../utils/foodDatabase';

export type RootStackParamList = {
  MainTabs: undefined;
  FoodSearch: {
    onSelectFood: (food: USDAFoodItem) => void;
  };
  CreateCustomMeal: {
    onSave?: (meal: USDAFoodItem) => void;
  };
  Settings: undefined;
};

export type MainTabParamList = {
  HomeTab: undefined;
  WorkoutsTab: undefined;
  NutritionTab: undefined;
  ProfileTab: undefined;
}; 