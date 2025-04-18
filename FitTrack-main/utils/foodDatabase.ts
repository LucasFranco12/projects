// API configuration
const USDA_API_KEY = 'z7mSupGKO0ZKDGFeFWiE87dCtZybKq00nHxLA8P5'; // Replace with your actual API key
const USDA_API_BASE_URL = 'https://api.nal.usda.gov/fdc/v1';

export interface USDAFoodItem {
  fdcId: string;
  description: string;
  brandName?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  nutrients: {
    protein: number;
    carbohydrates: number;
    fat: number;
    calories: number;
  };
  instructions?: string;
  includedFoods?: USDAFoodItem[];
}

export const searchFoods = async (query: string): Promise<USDAFoodItem[]> => {
  try {
    console.log('Searching foods with query:', query);
    const url = `${USDA_API_BASE_URL}/foods/search?api_key=${USDA_API_KEY}&query=${encodeURIComponent(query)}&pageSize=25`;
    console.log('API URL:', url);

    const response = await fetch(
      url,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('API Response:', data);
    
    if (!data || !data.foods || !Array.isArray(data.foods)) {
      console.log('Invalid response format:', data);
      return [];
    }

    const mappedFoods = data.foods.map((food: any) => {
      // Find nutrient values by nutrientName instead of nutrientId
      const findNutrientByName = (name: string) => {
        const nutrient = food.foodNutrients?.find((n: any) => 
          n.nutrientName?.toLowerCase().includes(name.toLowerCase())
        );
        return nutrient ? nutrient.value : 0;
      };

      // Get nutrient values
      const protein = findNutrientByName('protein');
      const carbs = findNutrientByName('carbohydrate');
      const fat = findNutrientByName('total fat');
      const calories = findNutrientByName('energy');

      return {
        fdcId: food.fdcId?.toString() || '',
        description: food.description || food.lowercaseDescription || 'Unknown Food',
        brandName: food.brandName,
        servingSize: food.servingSize,
        servingSizeUnit: food.servingSizeUnit,
        nutrients: {
          protein: Number(protein) || 0,
          carbohydrates: Number(carbs) || 0,
          fat: Number(fat) || 0,
          calories: Number(calories) || 0,
        }
      };
    });
    
    console.log('Mapped foods with nutrients:', mappedFoods);
    return mappedFoods;
  } catch (error) {
    console.error('Error fetching foods:', error);
    return [];
  }
}; 