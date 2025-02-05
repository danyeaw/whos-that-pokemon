import asyncio
import json
from pathlib import Path

import cv2
import httpx
import numpy as np

from pyscript.card_matcher import HASH_SIZE

BASE_URL = "https://api.pokemontcg.io/v2"


async def get_stellar_crown_data(api_key: str | None = None) -> list[dict]:
    """Get all cards from Stellar Crown set (SV07)"""
    headers = {"X-Api-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(headers=headers) as client:
        response = await client.get(
            f"{BASE_URL}/cards",
            params={"q": "set.id:sv7", "orderBy": "number"},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"]


def compute_average_hash(image) -> str:
    """Compute average hash using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE))
    avg_pixel = np.mean(resized)
    diff = resized > avg_pixel
    hash_str = "".join(["1" if b else "0" for b in diff.flatten()])
    return hex(int(hash_str, 2))[2:].zfill(16)


def compute_difference_hash(image) -> str:
    """Compute difference hash using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))
    diff = resized[:, 1:] > resized[:, :-1]
    hash_str = "".join(["1" if b else "0" for b in diff.flatten()])
    return hex(int(hash_str, 2))[2:].zfill(16)


def compute_image_hash(img_path: Path) -> str | None:
    """Compute both average and difference hashes for an image"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            return None

        avg_hash = compute_average_hash(img)
        dhash = compute_difference_hash(img)

        if avg_hash is None or dhash is None:
            return None

        return f"{avg_hash}:{dhash}"

    except (OSError, IOError) as e:
        print(f"Error reading image file {img_path}: {str(e)}")
        return None
    except cv2.error as e:
        print(f"OpenCV error processing {img_path}: {str(e)}")
        return None


def match_local_images(cards: list[dict], images_dir: Path) -> list[dict]:
    """Match downloaded images with card data and compute hashes"""
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
            image_hash = compute_image_hash(image_path)
            if image_hash:
                card["image_hash"] = image_hash
                print(f"Processed card {card_num}")
            else:
                print(f"Failed to hash for card {card_num}")
        else:
            print(f"No image found for card {card_num}")

    return cards


# ... rest of the code remains the same ...


def create_database(cards: list[dict], output_path: Path):
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
            card.get("tcgplayer", {}).get("prices", {}).get("normal", {}).get("market")
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

    minified_output = output_path.with_suffix(".min.json")
    with open(minified_output, "w", encoding="utf-8") as f:
        json.dump(database, f, separators=(",", ":"))

    # Print statistics
    print_database_statistics(database)


def print_database_statistics(database: dict):
    """Print statistics about the database"""
    cards = database["cards"]
    total_cards = len(cards)
    cards_with_local = len([c for c in cards if "local_image" in c])
    cards_with_api = len(
        [c for c in cards if c["images"]["small"] or c["images"]["large"]]
    )
    cards_with_prices = len(
        [
            c
            for c in cards
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

    # Get card data from Pokemon TCG API
    print("Fetching card data from Pokemon TCG API...")
    cards = await get_stellar_crown_data()

    # Match with local images and compute hashes
    print("Matching with downloaded card images...")
    cards = match_local_images(cards, images_dir)

    # Create database file
    print("Creating database file...")
    create_database(cards, output_path)

    print(f"\nDatabase created successfully at {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
