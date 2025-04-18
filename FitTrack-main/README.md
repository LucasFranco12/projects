# Fitness App - React Native with Expo Go

A comprehensive fitness application built with React Native and Expo, featuring workout tracking, nutrition monitoring, and an AI personal trainer.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Node.js** (v16.0.0 or higher)
   - Download from: [https://nodejs.org/](https://nodejs.org/)
   - Verify installation: `node --version`

2. **npm** (comes with Node.js)
   - Verify installation: `npm --version`

3. **Expo Go** app on your mobile device
   - [Android Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)
   - [iOS App Store](https://apps.apple.com/app/expo-go/id982107779)

## Installation

1. Clone the repository
```bash
git clone [your-repository-url]
cd FitnessAppExpo
```

2. Install dependencies
```bash
npm install
```

3. Install Python dependencies (for AI features) (RN not implemented but take a look at the research paper folder i collected over 200 papers doing with health and sicence so far!)
```bash
cd ai
pip install -r requirements.txt
cd ..
```

## Running the Application

1. Start the Expo development server
```bash
npx expo start
```

2. Connect your mobile device:
   - Ensure your phone and computer are on the same WiFi network
   - Open the Expo Go app on your phone
   - Scan the QR code displayed in your terminal
     - iOS users: Use your phone's camera
     - Android users: Use the Expo Go app's QR scanner

## Troubleshooting

If you encounter any issues:

1. Clear npm cache:
```bash
npm cache clean --force
```

2. Delete node_modules and reinstall:
```bash
rm -rf node_modules
npm install
```

3. Reset Expo cache:
```bash
npx expo start -c
```

## Common Issues

- **"Unable to find expo in this project"**: Run `npm install expo` to install Expo locally
- **QR Code not scanning**: Ensure both devices are on the same network
- **Metro bundler issues**: Try clearing the Metro cache with `npx expo start -c`

## Additional Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Native Documentation](https://reactnative.dev/docs/getting-started)
- [Node.js Documentation](https://nodejs.org/en/docs/)

## Project Structure
FitnessAppExpo/
├── ai/ # AI trainer scripts
├── assets/ # Images and static files
├── components/ # Reusable React components
├── navigation/ # Navigation configuration
├── screens/ # Application screens
├── utils/ # Utility functions
└── App.tsx # Root component

## Features

- Workout tracking
- Nutrition monitoring
- AI personal trainer
- Progress tracking
- Custom meal planning
- Exercise library

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

