import cv2
import numpy as np
import imagehash
from PIL import Image
import json
from pathlib import Path

class CardMatcher:
    def __init__(self, debug_log):
        self.db = None
        self.cards = []
        self.debug_log = debug_log
        self.load_database()

    def load_database(self):
        """Load and validate the database"""
        try:
            db_path = Path('pokemon_cards.json')

            if not db_path.exists():
                self.debug_log(f"Error: Database file not found at {db_path}")
                return False

            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict) or 'cards' not in data:
                self.debug_log("Error: Invalid database structure")
                return False

            self.cards = data['cards']

            if not self.cards:
                self.debug_log("Error: Database is empty")
                return False

            return True

        except Exception as e:
            self.debug_log(f"Error loading database: {str(e)}")
            return False

    def compute_image_hash(self, img_array):
        """Compute hash string for a cv2/numpy image array"""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # Use the same hash size as in database
            avg_hash = str(imagehash.average_hash(pil_image))
            phash = str(imagehash.phash(pil_image))
            dhash = str(imagehash.dhash(pil_image))
            colorhash = str(imagehash.colorhash(pil_image))

            # Return hash string in same format as database
            return f"{avg_hash}:{phash}:{dhash}:{colorhash}"

        except Exception as e:
            self.debug_log(f"Error computing hash: {str(e)}")
            return None

    def safe_hex_to_hash(self, hex_str):
        """Safely convert hex string to hash object"""
        try:
            return imagehash.hex_to_hash(hex_str)
        except Exception as e:
            self.debug_log(f"Error converting hash {hex_str}: {str(e)}")
            return None

    def hash_difference(self, hash1, hash2):
        """Compare two hash strings with weights for all four hash types"""
        try:
            # Split the hashes
            h1_parts = hash1.split(':')
            h2_parts = hash2.split(':')

            # Verify we have all four hash parts
            if len(h1_parts) != 4 or len(h2_parts) != 4:
                self.debug_log(f"Hash format mismatch: {len(h1_parts)} vs {len(h2_parts)} parts")
                return float('inf')

            # Weights for each hash type
            weights = {
                'average': 0.40,    # Overall structure
                'perceptual': 0.35, # Fine details
                'difference': 0.25, # Gradients
                'color': 0       # Color information
            }

            # Calculate weighted differences
            total_diff = 0

            # Handle first three hashes (average, perceptual, difference)
            for i in range(3):
                hash1_obj = imagehash.hex_to_hash(h1_parts[i])
                hash2_obj = imagehash.hex_to_hash(h2_parts[i])
                diff = float(hash1_obj - hash2_obj)
                hash_type = ['average', 'perceptual', 'difference'][i]
                weighted_diff = diff * weights[hash_type]
                total_diff += weighted_diff

            # Handle colorhash separately
            color_diff = float(abs(int(h1_parts[3], 16) - int(h2_parts[3], 16)))
            # Normalize color difference to be in similar range as other diffs
            color_diff = color_diff / (2**32) * 64.0  # Scale to roughly 0-64 range
            weighted_color_diff = color_diff * weights['color']
            total_diff += weighted_color_diff

            return total_diff

        except Exception as e:
            self.debug_log(f"Error comparing hashes: {str(e)}")
            import traceback
            self.debug_log(traceback.format_exc())
            return float('inf')

    def find_matching_card(self, img_array, threshold=15.0):
        """Find best matching card"""
        try:
            if not self.cards:
                self.debug_log("Error: No cards in database")
                return None

            card_hash = self.compute_image_hash(img_array)
            if not card_hash:
                self.debug_log("Error: Failed to compute hash for detected card")
                return None

            best_match = None
            min_diff = float('inf')
            all_matches = []

            for card in self.cards:
                if 'image_hash' not in card:
                    continue

                diff = self.hash_difference(card_hash, card['image_hash'])
                if diff != float('inf'):
                    all_matches.append((card, diff))
                    if diff < min_diff:
                        min_diff = diff
                        best_match = card

            # Print top matches
            all_matches.sort(key=lambda x: x[1])
            self.debug_log("\nTop 3 matches:")
            for card, diff in all_matches[:3]:
                self.debug_log(f"{card['name']} (#{card['number']}): diff = {diff:.2f}")

            if best_match:
                confidence = max(0, min(1.0, 1.0 - (min_diff / threshold)))
                match_quality = "High" if min_diff < threshold else "Low"

                return {
                    'name': best_match['name'],
                    'number': best_match['number'],
                    'rarity': best_match.get('rarity', 'Unknown'),
                    'supertype': best_match.get('supertype', 'Unknown'),
                    'subtypes': best_match.get('subtypes', []),
                    'images': best_match.get('images', []),
                    'market_prices': best_match.get('market_prices'),
                    'confidence': confidence,
                    'difference': min_diff,
                    'match_quality': match_quality,
                }

            return None

        except Exception as e:
            self.debug_log(f"Error finding match: {str(e)}")
            import traceback
            self.debug_log(traceback.format_exc())
            return None
