from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List
import datetime


class DocumentLoader:
    """
    Loads documents from a directory, handling multiple file types.
    Attaches rich metadata to every document.
    """

    LOADERS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def load_directory(self, directory: str) -> List[Document]:
        """
        Load documents from a directory, supporting multiple file types.

        Args:
            directory (str): Path to the directory containing documents.
        Returns:
            List[Document]: A list of loaded documents with metadata.
        """
        all_docs = []
        path = Path(directory)

        for file_path in path.rglob("*"):
            suffix = file_path.suffix.lower()

            if suffix not in self.LOADERS:
                continue  # skip unsupported files
            try:
                loader_class = self.LOADERS[suffix]
                loader = loader_class(str(file_path))
                docs = loader.load()

                # Attach production metadata to every doc
                for doc in docs:
                    doc.metadata.update(
                        {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": suffix,
                            "ingested_at": datetime.datetime.utcnow().isoformat(),
                        }
                    )

                all_docs.extend(docs)
                print(f"  📄 Loaded: {file_path.name} ({len(docs)} pages)")

            except Exception as e:
                # In production, never crash on a single bad file
                print(f"  ⚠️  Failed to load {file_path.name}: {e}")
                continue

        print(f"\n✅ Total documents loaded: {len(all_docs)}")
        return all_docs
