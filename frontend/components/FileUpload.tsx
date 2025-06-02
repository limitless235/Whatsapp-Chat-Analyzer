import React from "react";

export type FileUploadProps = {
  onFileSelect: (file: File) => void;
};

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="mt-4">
      <input type="file" accept=".txt" onChange={handleChange} />
    </div>
  );
};

export default FileUpload;
