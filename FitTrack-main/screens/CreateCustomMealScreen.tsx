import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Modal,
  ActivityIndicator
} from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { Ionicons } from '@expo/vector-icons';
import { USDAFoodItem, searchFoods } from '../utils/foodDatabase';
import FoodItem from '../components/FoodItem';

interface CreateCustomMealScreenProps {
  navigation: any;
  route: {
    params: {
      onSave: (meal: USDAFoodItem) => void;
    };
  };
}

const CreateCustomMealScreen = ({ navigation, route }: CreateCustomMealScreenProps) => {
  const [name, setName] = useState('');
  const [servingSize, setServingSize] = useState('');
  const [servingUnit, setServingUnit] = useState('g');
  const [protein, setProtein] = useState('');
  const [carbs, setCarbs] = useState('');
  const [fat, setFat] = useState('');
  const [instructions, setInstructions] = useState('');
  const [showFoodSearch, setShowFoodSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<USDAFoodItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [includedFoods, setIncludedFoods] = useState<USDAFoodItem[]>([]);

  const calculateCalories = () => {
    const p = parseFloat(protein) || 0;
    const c = parseFloat(carbs) || 0;
    const f = parseFloat(fat) || 0;
    return (p * 4) + (c * 4) + (f * 9);
  };

  const handleSearch = async () => {
    if (searchQuery.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    setSearchResults([]);

    try {
      const results = await searchFoods(searchQuery);
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      Alert.alert('Error', 'Failed to search for foods. Please try again.');
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const addFoodToMeal = (food: USDAFoodItem) => {
    setIncludedFoods(prev => [...prev, food]);
    
    // Update total macros
    const newProtein = (parseFloat(protein) || 0) + food.nutrients.protein;
    const newCarbs = (parseFloat(carbs) || 0) + food.nutrients.carbohydrates;
    const newFat = (parseFloat(fat) || 0) + food.nutrients.fat;
    
    setProtein(newProtein.toFixed(1));
    setCarbs(newCarbs.toFixed(1));
    setFat(newFat.toFixed(1));
    
    setShowFoodSearch(false);
    setSearchQuery('');
    setSearchResults([]);
  };

  const removeFood = (food: USDAFoodItem) => {
    setIncludedFoods(prev => prev.filter(f => f.fdcId !== food.fdcId));
    
    // Update total macros
    const newProtein = (parseFloat(protein) || 0) - food.nutrients.protein;
    const newCarbs = (parseFloat(carbs) || 0) - food.nutrients.carbohydrates;
    const newFat = (parseFloat(fat) || 0) - food.nutrients.fat;
    
    setProtein(Math.max(0, newProtein).toFixed(1));
    setCarbs(Math.max(0, newCarbs).toFixed(1));
    setFat(Math.max(0, newFat).toFixed(1));
  };

  const handleSave = () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter a meal name');
      return;
    }

    if (!protein && !carbs && !fat) {
      Alert.alert('Error', 'Please enter at least one macro nutrient');
      return;
    }

    const customMeal: USDAFoodItem = {
      fdcId: `custom_${Date.now()}`,
      description: name.trim(),
      servingSize: parseFloat(servingSize) || undefined,
      servingSizeUnit: servingUnit,
      nutrients: {
        protein: parseFloat(protein) || 0,
        carbohydrates: parseFloat(carbs) || 0,
        fat: parseFloat(fat) || 0,
        calories: calculateCalories(),
      },
      instructions: instructions.trim() || undefined,
      includedFoods: includedFoods.length > 0 ? includedFoods : undefined
    };

    route.params.onSave(customMeal);
    navigation.goBack();
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
        <Text style={styles.headerTitle}>Create Custom Meal</Text>
      </View>

      <ScrollView style={styles.content}>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Meal Name</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="Enter meal name"
            placeholderTextColor={COLORS.subtext}
          />
        </View>

        <View style={styles.row}>
          <View style={[styles.inputGroup, { flex: 2 }]}>
            <Text style={styles.label}>Serving Size</Text>
            <TextInput
              style={styles.input}
              value={servingSize}
              onChangeText={setServingSize}
              placeholder="Amount"
              keyboardType="numeric"
              placeholderTextColor={COLORS.subtext}
            />
          </View>

          <View style={[styles.inputGroup, { flex: 1 }]}>
            <Text style={styles.label}>Unit</Text>
            <TextInput
              style={styles.input}
              value={servingUnit}
              onChangeText={setServingUnit}
              placeholder="g"
              placeholderTextColor={COLORS.subtext}
            />
          </View>
        </View>

        <TouchableOpacity
          style={styles.searchButton}
          onPress={() => setShowFoodSearch(true)}
        >
          <Ionicons name="search" size={20} color={COLORS.card} />
          <Text style={styles.searchButtonText}>Search Foods to Add</Text>
        </TouchableOpacity>

        {includedFoods.length > 0 && (
          <View style={styles.includedFoodsSection}>
            <Text style={styles.sectionTitle}>Included Foods</Text>
            {includedFoods.map((food, index) => (
              <View key={`${food.fdcId}-${index}`} style={styles.includedFoodItem}>
                <Text style={styles.includedFoodName}>{food.description}</Text>
                <TouchableOpacity
                  style={styles.removeButton}
                  onPress={() => removeFood(food)}
                >
                  <Ionicons name="close-circle" size={24} color={COLORS.error} />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}

        <Text style={styles.sectionTitle}>Macronutrients</Text>

        <View style={styles.macroInputs}>
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Protein (g)</Text>
            <TextInput
              style={styles.input}
              value={protein}
              onChangeText={setProtein}
              keyboardType="numeric"
              placeholder="0"
              placeholderTextColor={COLORS.subtext}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Carbs (g)</Text>
            <TextInput
              style={styles.input}
              value={carbs}
              onChangeText={setCarbs}
              keyboardType="numeric"
              placeholder="0"
              placeholderTextColor={COLORS.subtext}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Fat (g)</Text>
            <TextInput
              style={styles.input}
              value={fat}
              onChangeText={setFat}
              keyboardType="numeric"
              placeholder="0"
              placeholderTextColor={COLORS.subtext}
            />
          </View>
        </View>

        <View style={styles.calorieDisplay}>
          <Text style={styles.calorieLabel}>Calculated Calories:</Text>
          <Text style={styles.calorieValue}>{calculateCalories()} kcal</Text>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Instructions (Optional)</Text>
          <TextInput
            style={[styles.input, styles.instructionsInput]}
            value={instructions}
            onChangeText={setInstructions}
            placeholder="Add preparation instructions..."
            placeholderTextColor={COLORS.subtext}
            multiline
            numberOfLines={4}
            textAlignVertical="top"
          />
        </View>
      </ScrollView>

      <TouchableOpacity
        style={styles.saveButton}
        onPress={handleSave}
      >
        <Text style={styles.saveButtonText}>Save Meal</Text>
      </TouchableOpacity>

      <Modal
        visible={showFoodSearch}
        animationType="slide"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Search Foods</Text>
              <TouchableOpacity 
                onPress={() => {
                  setShowFoodSearch(false);
                  setSearchQuery('');
                  setSearchResults([]);
                }}
              >
                <Ionicons name="close" size={24} color={COLORS.text} />
              </TouchableOpacity>
            </View>

            <View style={styles.searchInputContainer}>
              <TextInput
                style={styles.searchInput}
                value={searchQuery}
                onChangeText={setSearchQuery}
                placeholder="Search for a food..."
                placeholderTextColor={COLORS.subtext}
                onSubmitEditing={handleSearch}
                returnKeyType="search"
              />
              <TouchableOpacity
                style={[styles.searchActionButton, searchQuery.length < 2 && styles.searchButtonDisabled]}
                onPress={handleSearch}
                disabled={searchQuery.length < 2}
              >
                <Text style={styles.searchActionButtonText}>Search</Text>
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.searchResults}>
              {isSearching ? (
                <ActivityIndicator size="large" color={COLORS.primary} />
              ) : (
                searchResults.map((food) => (
                  <FoodItem
                    key={food.fdcId}
                    food={food}
                    onSelect={() => addFoodToMeal(food)}
                    onFavorite={() => {}}
                    isFavorite={false}
                  />
                ))
              )}
              {!isSearching && searchResults.length === 0 && searchQuery.length >= 2 && (
                <Text style={styles.noResultsText}>No foods found</Text>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
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
  content: {
    flex: 1,
    padding: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: SIZES.small,
    color: COLORS.text,
    marginBottom: 8,
  },
  input: {
    backgroundColor: COLORS.card,
    borderRadius: 8,
    padding: 12,
    fontSize: SIZES.medium,
    color: COLORS.text,
    ...SHADOWS.small,
  },
  instructionsInput: {
    height: 100,
    textAlignVertical: 'top',
  },
  row: {
    flexDirection: 'row',
    gap: 16,
  },
  sectionTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: 16,
    marginBottom: 16,
  },
  macroInputs: {
    gap: 16,
  },
  calorieDisplay: {
    backgroundColor: COLORS.card,
    padding: 16,
    borderRadius: 8,
    marginTop: 24,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    ...SHADOWS.small,
  },
  calorieLabel: {
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  calorieValue: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  saveButton: {
    backgroundColor: COLORS.primary,
    margin: 16,
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    ...SHADOWS.medium,
  },
  saveButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  searchButton: {
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    gap: 8,
  },
  searchButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
  },
  modalContent: {
    backgroundColor: COLORS.background,
    flex: 1,
    marginTop: 60,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  searchInputContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderRadius: 8,
    padding: 12,
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  searchActionButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 8,
    padding: 12,
    justifyContent: 'center',
    minWidth: 80,
    alignItems: 'center',
  },
  searchActionButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  searchButtonDisabled: {
    backgroundColor: COLORS.subtext,
  },
  searchResults: {
    flex: 1,
  },
  noResultsText: {
    textAlign: 'center',
    color: COLORS.subtext,
    fontSize: SIZES.medium,
    marginTop: 20,
  },
  includedFoodsSection: {
    marginBottom: 16,
  },
  includedFoodItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.card,
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  includedFoodName: {
    flex: 1,
    fontSize: SIZES.medium,
    color: COLORS.text,
    marginRight: 8,
  },
  removeButton: {
    padding: 4,
  },
});

export default CreateCustomMealScreen; 