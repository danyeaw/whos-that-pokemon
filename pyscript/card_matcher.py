import cv2
import numpy as np
import json
from pathlib import Path

HASH_SIZE = 8

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

    def compute_average_hash(self, image):
        """Compute average hash using OpenCV"""
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE))

            # Calculate average pixel value
            avg_pixel = np.mean(resized)

            # Convert to binary hash string
            diff = resized > avg_pixel
            # Convert boolean array to hash string
            hash_str = ''.join(['1' if b else '0' for b in diff.flatten()])

            # Convert binary string to hexadecimal
            hash_hex = hex(int(hash_str, 2))[2:].zfill(16)

            return hash_hex

        except Exception as e:
            self.debug_log(f"Error computing average hash: {str(e)}")
            return None

    def compute_difference_hash(self, image):
        """Compute difference hash using OpenCV"""
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))

            # Compute differences
            diff = resized[:, 1:] > resized[:, :-1]

            # Convert to hash string
            hash_str = ''.join(['1' if b else '0' for b in diff.flatten()])

            # Convert binary string to hexadecimal
            hash_hex = hex(int(hash_str, 2))[2:].zfill(16)

            return hash_hex

        except Exception as e:
            self.debug_log(f"Error computing difference hash: {str(e)}")
            return None

    def compute_image_hash(self, img_array):
        """Compute both average and difference hashes for an image"""
        try:
            avg_hash = self.compute_average_hash(img_array)
            dhash = self.compute_difference_hash(img_array)

            if avg_hash is None or dhash is None:
                return None

            return f"{avg_hash}:{dhash}"

        except Exception as e:
            self.debug_log(f"Error computing hash: {str(e)}")
            return None

    def hamming_distance(self, hash1_hex, hash2_hex):
        """Calculate Hamming distance between two hex hashes"""
        try:
            # Convert hex strings to binary strings
            hash1_bin = bin(int(hash1_hex, 16))[2:].zfill(64)
            hash2_bin = bin(int(hash2_hex, 16))[2:].zfill(64)

            # Calculate Hamming distance
            return sum(c1 != c2 for c1, c2 in zip(hash1_bin, hash2_bin))

        except Exception as e:
            self.debug_log(f"Error calculating Hamming distance: {str(e)}")
            return float('inf')

    def hash_difference(self, hash1, hash2):
        """Compare two hash strings with weights for average and difference hashes"""
        try:
            # Split the hashes
            h1_parts = hash1.split(':')
            h2_parts = hash2.split(':')

            # Verify we have both hash parts
            if len(h1_parts) != 2 or len(h2_parts) != 2:
                self.debug_log(f"Hash format mismatch: {len(h1_parts)} vs {len(h2_parts)} parts")
                return float('inf')

            # Weights for each hash type
            weights = {
                'average': 0.6,     # Overall structure
                'difference': 0.4    # Gradients
            }

            # Calculate weighted Hamming distances
            avg_dist = self.hamming_distance(h1_parts[0], h2_parts[0])
            diff_dist = self.hamming_distance(h1_parts[1], h2_parts[1])

            # Combine weighted distances
            total_diff = (avg_dist * weights['average'] +
                          diff_dist * weights['difference'])

            return total_diff

        except Exception as e:
            self.debug_log(f"Error comparing hashes: {str(e)}")
            import traceback
            self.debug_log(traceback.format_exc())
            return float('inf')

    def find_matching_card(self, img_array, threshold=24.0):
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
                # Return to original confidence calculation
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
