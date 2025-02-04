import httpx
from pathlib import Path
import asyncio

BASE_URL = "https://dz3we2x72f7ol.cloudfront.net/expansions/stellar-crown/en-us/"
MAX_CARD_NUMBER = 175


async def download_cards():
    # Create images directory
    images_dir = Path("card_images")
    images_dir.mkdir(exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
        tasks = []
        for i in range(1, MAX_CARD_NUMBER + 1):
            url = f"{BASE_URL}SV07_EN_{i}-2x.png"

            filename = Path(f"card_images/Stellar-Crown_{str(i).zfill(3)}.png")
            task = asyncio.create_task(download_card(client, url, filename, i))
            tasks.append(task)

        # Download in batches of 5 to be nice to the server
        for batch in chunks(tasks, 5):
            await asyncio.gather(*batch)
            await asyncio.sleep(1)  # Rate limiting delay between batches


async def download_card(
    client: httpx.AsyncClient, url: str, filepath: Path, card_num: int
):
    try:
        response = await client.get(url)
        response.raise_for_status()

        # Check if we actually got an image
        if "image" in response.headers.get("content-type", ""):
            filepath.write_bytes(response.content)
            print(f"Downloaded card {card_num}: {filepath.name}")
        else:
            print(f"No card found at {url}")

    except httpx.HTTPError as e:
        if (
            e.response.status_code != 404
        ):  # Ignore 404s as we expect some numbers to not exist
            print(f"Error downloading card {card_num}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error downloading card {card_num}: {str(e)}")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    asyncio.run(download_cards())
