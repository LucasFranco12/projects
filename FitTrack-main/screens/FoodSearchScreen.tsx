import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
  Alert
} from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { Ionicons } from '@expo/vector-icons';
import { searchFoods, USDAFoodItem } from '../utils/foodDatabase';
import AsyncStorage from '@react-native-async-storage/async-storage';
import FoodItem from '../components/FoodItem';

interface FoodSearchScreenProps {
  navigation: any;
  route: {
    params: {
      onSelectFood: (food: USDAFoodItem) => void;
    };
  };
}

const FoodSearchScreen = ({ navigation, route }: FoodSearchScreenProps) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<USDAFoodItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [recentFoods, setRecentFoods] = useState<USDAFoodItem[]>([]);
  const [favoriteFoods, setFavoriteFoods] = useState<USDAFoodItem[]>([]);

  useEffect(() => {
    loadSavedFoods();
  }, []);

  const loadSavedFoods = async () => {
    try {
      const recentFoodsJson = await AsyncStorage.getItem('recentFoods');
      const favoriteFoodsJson = await AsyncStorage.getItem('favoriteFoods');
      
      if (recentFoodsJson) {
        setRecentFoods(JSON.parse(recentFoodsJson));
      }
      if (favoriteFoodsJson) {
        setFavoriteFoods(JSON.parse(favoriteFoodsJson));
      }
    } catch (error) {
      console.error('Error loading saved foods:', error);
    }
  };

  const handleSearch = async () => {
    if (searchQuery.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    setSearchResults([]);

    try {
      console.log('Searching for:', searchQuery);
      const results = await searchFoods(searchQuery);
      console.log('Search results:', results);
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      Alert.alert('Error', 'Failed to search for foods. Please try again.');
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleFoodSelect = async (food: USDAFoodItem) => {
    try {
      const updatedRecentFoods = [food, ...recentFoods.filter(f => f.fdcId !== food.fdcId)].slice(0, 5);
      await AsyncStorage.setItem('recentFoods', JSON.stringify(updatedRecentFoods));
      setRecentFoods(updatedRecentFoods);
      route.params.onSelectFood(food);
      navigation.goBack();
    } catch (error) {
      console.error('Error saving recent food:', error);
    }
  };

  const toggleFavorite = async (food: USDAFoodItem) => {
    try {
      const isFavorite = favoriteFoods.some(f => f.fdcId === food.fdcId);
      let updatedFavorites;
      
      if (isFavorite) {
        updatedFavorites = favoriteFoods.filter(f => f.fdcId !== food.fdcId);
      } else {
        updatedFavorites = [...favoriteFoods, food];
      }
      
      await AsyncStorage.setItem('favoriteFoods', JSON.stringify(updatedFavorites));
      setFavoriteFoods(updatedFavorites);
    } catch (error) {
      console.error('Error updating favorites:', error);
    }
  };

  const navigateToCreateCustomMeal = () => {
    navigation.navigate('CreateCustomMeal', {
      onSave: (customMeal: USDAFoodItem) => {
        handleFoodSelect(customMeal);
      },
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color={COLORS.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Search Foods</Text>
      </View>

      <View style={styles.searchContainer}>
        <View style={styles.searchInputContainer}>
          <TextInput
            style={styles.searchInput}
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search for a food..."
            placeholderTextColor={COLORS.subtext}
            returnKeyType="search"
            onSubmitEditing={handleSearch}
          />
          <TouchableOpacity
            style={[
              styles.searchButton,
              searchQuery.length < 2 && styles.searchButtonDisabled
            ]}
            onPress={handleSearch}
            disabled={searchQuery.length < 2}
          >
            <Text style={styles.searchButtonText}>Search</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={styles.createCustomButton}
          onPress={navigateToCreateCustomMeal}
        >
          <Ionicons name="add-circle-outline" size={24} color={COLORS.primary} />
          <Text style={styles.createCustomButtonText}>Create Custom Meal</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {isSearching ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.loadingText}>Searching...</Text>
          </View>
        ) : (
          <>
            {searchResults.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Search Results</Text>
                {searchResults.map((food) => (
                  <FoodItem
                    key={food.fdcId}
                    food={food}
                    onSelect={() => handleFoodSelect(food)}
                    onFavorite={() => toggleFavorite(food)}
                    isFavorite={favoriteFoods.some(f => f.fdcId === food.fdcId)}
                  />
                ))}
              </View>
            )}

            {searchResults.length === 0 && searchQuery.length >= 2 && !isSearching && (
              <View style={styles.noResultsContainer}>
                <Text style={styles.noResultsText}>No foods found</Text>
              </View>
            )}

            {favoriteFoods.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Favorite Foods</Text>
                {favoriteFoods.map((food) => (
                  <FoodItem
                    key={food.fdcId}
                    food={food}
                    onSelect={() => handleFoodSelect(food)}
                    onFavorite={() => toggleFavorite(food)}
                    isFavorite={true}
                  />
                ))}
              </View>
            )}

            {recentFoods.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Recent Foods</Text>
                {recentFoods.map((food) => (
                  <FoodItem
                    key={food.fdcId}
                    food={food}
                    onSelect={() => handleFoodSelect(food)}
                    onFavorite={() => toggleFavorite(food)}
                    isFavorite={favoriteFoods.some(f => f.fdcId === food.fdcId)}
                  />
                ))}
              </View>
            )}
          </>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: COLORS.card,
    ...SHADOWS.medium,
  },
  backButton: {
    marginRight: 16,
  },
  headerTitle: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  searchContainer: {
    padding: 16,
    backgroundColor: COLORS.card,
    ...SHADOWS.small,
  },
  searchInputContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  searchInput: {
    flex: 1,
    backgroundColor: COLORS.background,
    borderRadius: 8,
    padding: 12,
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  searchButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 8,
    paddingHorizontal: 16,
    justifyContent: 'center',
    alignItems: 'center',
    minWidth: 100,
  },
  searchButtonDisabled: {
    backgroundColor: COLORS.subtext,
  },
  searchButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  createCustomButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
    padding: 12,
    backgroundColor: COLORS.background,
    borderRadius: 8,
    gap: 8,
  },
  createCustomButtonText: {
    color: COLORS.primary,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 12,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loadingText: {
    marginTop: 12,
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  noResultsContainer: {
    padding: 24,
    alignItems: 'center',
  },
  noResultsText: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
  },
});

export default FoodSearchScreen; 