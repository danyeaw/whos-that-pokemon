import httpx
import asyncio
from pathlib import Path
import json
import cv2
import numpy as np


class PokemonTCGDatabase:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.base_url = "https://api.pokemontcg.io/v2"
        self.headers = {"X-Api-Key": api_key} if api_key else {}

    async def get_stellar_crown_data(self) -> list[dict]:
        """Get all cards from Stellar Crown set (SV07)"""
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(
                f"{self.base_url}/cards",
                params={"q": "set.id:sv7", "orderBy": "number"},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"]

    def compute_image_hash(self, img_path: Path) -> str | None:
        """Compute perceptual hash using DCT"""
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize to 32x32 and compute DCT
            size = 32
            gray_resized = cv2.resize(gray, (size, size))
            dct = cv2.dct(np.float32(gray_resized))

            # Use top-left 8x8 of DCT
            dct_low = dct[:8, :8]
            avg_dct = dct_low.mean()

            # Convert to binary string first
            binary_hash = "".join(
                ["1" if c >= avg_dct else "0" for c in dct_low.flatten()]
            )
            # Convert to hexadecimal
            hex_hash = hex(int(binary_hash, 2))[2:].zfill(16)  # 64 bits = 16 hex chars
            return hex_hash

        except Exception as e:
            print(f"Error computing hash for {img_path}: {str(e)}")
            return None

    def match_local_images(self, cards: list[dict], images_dir: Path) -> list[dict]:
        """Match downloaded images with card data and compute hashes"""
        # Get all image files
        image_files = list(images_dir.glob("*.png"))

        # Create number to image mapping
        image_map = {}
        for img_path in image_files:
            match = img_path.stem.split("_")[1]  # Get number from filename
            if match:
                image_map[match] = img_path

        # Match images with card data
        for card in cards:
            card_num = card.get("number", "").zfill(3)

            if card_num in image_map:
                image_path = image_map[card_num]
                card["local_image"] = str(image_path)

                # Compute image hash
                image_hash = self.compute_image_hash(image_path)
                if image_hash:
                    card["image_hash"] = image_hash
                    print(f"Processed card {card_num}")
                else:
                    print(f"Failed to compute hash for card {card_num}")
            else:
                print(f"No image found for card {card_num}")

        return cards

    def create_database(self, cards: list[dict], output_path: Path):
        """Create the final database file"""
        database = {
            "set_info": {
                "name": "Stellar Crown",
                "series": "Scarlet & Violet",
                "set_code": "SV07",
            },
            "cards": [],
        }

        for card in cards:
            # Extract just the market prices
            tcgplayer_market = (
                card.get("tcgplayer", {})
                .get("prices", {})
                .get("normal", {})
                .get("market")
            )
            cardmarket_price = card.get("cardmarket", {}).get("prices", {}).get("avg30")

            db_card = {
                "name": card.get("name"),
                "number": card.get("number"),
                "supertype": card.get("supertype"),
                "subtypes": card.get("subtypes", []),
                "rarity": card.get("rarity"),
                "images": {
                    "small": card.get("images", {}).get("small"),
                    "large": card.get("images", {}).get("large"),
                },
                "market_prices": {
                    "tcgplayer": tcgplayer_market,
                    "cardmarket": cardmarket_price,
                    "updated_at": max(
                        card.get("tcgplayer", {}).get("updatedAt", ""),
                        card.get("cardmarket", {}).get("updatedAt", ""),
                    )
                    or None,
                },
            }

            if card.get("local_image"):
                db_card["local_image"] = card.get("local_image")
            if card.get("image_hash"):
                db_card["image_hash"] = card.get("image_hash")

            database["cards"].append(db_card)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2)

        # Print statistics
        total_cards = len(cards)
        cards_with_local = len([c for c in database["cards"] if "local_image" in c])
        cards_with_api = len(
            [
                c
                for c in database["cards"]
                if c["images"]["small"] or c["images"]["large"]
            ]
        )
        cards_with_prices = len(
            [
                c
                for c in database["cards"]
                if c["market_prices"]["tcgplayer"] is not None
                or c["market_prices"]["cardmarket"] is not None
            ]
        )

        print("\nDatabase Statistics:")
        print(f"Total cards in set: {total_cards}")
        print(f"Cards with local images and hashes: {cards_with_local}")
        print(f"Cards with API images: {cards_with_api}")
        print(f"Cards with market prices: {cards_with_prices}")
        print(f"Cards missing local images: {total_cards - cards_with_local}")
        print(f"Cards missing API images: {total_cards - cards_with_api}")
        print(f"Cards missing prices: {total_cards - cards_with_prices}")


async def main():
    # Configuration
    images_dir = Path("card_images")
    output_path = Path("pokemon_cards.json")

    # Verify images directory exists
    if not images_dir.exists():
        print("Error: card_images directory not found!")
        return

    try:
        # Initialize database builder
        builder = PokemonTCGDatabase()

        # Get card data from Pokemon TCG API
        print("Fetching card data from Pokemon TCG API...")
        cards = await builder.get_stellar_crown_data()

        # Match with local images and compute hashes
        print("Matching with downloaded card images...")
        cards = builder.match_local_images(cards, images_dir)

        # Create database file
        print("Creating database file...")
        builder.create_database(cards, output_path)

        print(f"\nDatabase created successfully at {output_path}")

    except Exception as e:
        print(f"Error building database: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
