"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Upload, Loader2, Sparkles, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import Image from "next/image"

interface ClassificationResult {
  label: string
  confidence: number
}

export default function ImageClassifier() {
  const [image, setImage] = useState<string | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [results, setResults] = useState<ClassificationResult[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const processImage = async (file: File) => {
    const reader = new FileReader()
    reader.onload = async (e) => {
      const imageUrl = e.target?.result as string
      setImage(imageUrl)
      setResults([])

      // Simulate classification
      setIsClassifying(true)
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Mock results - Replace with actual API call
      const mockResults: ClassificationResult[] = [
        { label: "Golden Retriever", confidence: 0.94 },
        { label: "Labrador Retriever", confidence: 0.78 },
        { label: "Dog", confidence: 0.65 },
        { label: "Mammal", confidence: 0.52 },
        { label: "Pet", confidence: 0.48 },
      ]

      setResults(mockResults)
      setIsClassifying(false)
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) {
      await processImage(file)
    }
  }, [])

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      await processImage(file)
    }
  }

  const handleReset = () => {
    setImage(null)
    setResults([])
    setIsClassifying(false)
  }

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden border-2 border-border bg-card">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn("relative transition-colors", isDragging && "bg-primary/10 border-primary")}
        >
          {!image ? (
            <div className="flex flex-col items-center justify-center p-12 md:p-20">
              <div className="mb-6 rounded-full bg-primary/10 p-6">
                <ImageIcon className="h-12 w-12 text-primary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Upload an image</h3>
              <p className="mb-6 text-center text-sm text-muted-foreground text-pretty">
                Drag and drop an image here, or click to browse
              </p>
              <Button 
                className="cursor-pointer" 
                size="lg"
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <Upload className="mr-2 h-5 w-5" />
                Choose File
              </Button>
              <input 
                id="file-upload" 
                type="file" 
                accept="image/*" 
                className="hidden" 
                onChange={handleFileInput} 
              />
            </div>
          ) : (
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-center">Picture</h3>
                  <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-secondary">
                    <Image src={image || "/placeholder.svg"} alt="Original image" fill className="object-contain" />
                  </div>
                </div>
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-center">Area of Effect</h3>
                  <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-secondary">
                    <Image src={image || "/placeholder.svg"} alt="Area of effect" fill className="object-contain" />
                  </div>
                </div>
              </div>
              <div className="mt-4 flex justify-end">
                <Button variant="outline" onClick={handleReset}>
                  Upload New Image
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>

      {image && (
        <Card className="border-2 border-border bg-card p-6">
          <div className="mb-6 flex items-center gap-3">
            <div className="rounded-full bg-primary/10 p-2">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold">Classification Results</h2>
          </div>

          {isClassifying ? (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <p className="mt-4 text-muted-foreground">Analyzing image...</p>
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-4">
              {results.map((result, index) => (
                <div
                  key={index}
                  className="group overflow-hidden rounded-lg border border-border bg-secondary/50 p-4 transition-colors hover:bg-secondary"
                >
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                        {index + 1}
                      </span>
                      <span className="text-lg font-medium">{result.label}</span>
                    </div>
                    <span className="text-xl font-bold text-primary">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary transition-all duration-500 ease-out"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </Card>
      )}
    </div>
  )
}