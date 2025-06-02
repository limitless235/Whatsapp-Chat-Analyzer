"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";

type FileUploadProps = {
  onUpload: (file: File) => void;
};

const FileUpload: React.FC<FileUploadProps> = ({ onUpload }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      onUpload(file);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 rounded-xl">
      <input
        type="file"
        accept=".txt,.csv"
        onChange={handleFileChange}
        className="hidden"
        id="file-upload"
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-6 h-6 text-gray-600" />
          <span className="text-sm text-gray-500">
            Click to upload WhatsApp chat (.txt) or CSV
          </span>
        </div>
      </label>
      {selectedFile && (
        <p className="mt-2 text-xs text-gray-600">{selectedFile.name}</p>
      )}
    </div>
  );
};

export default FileUpload;
