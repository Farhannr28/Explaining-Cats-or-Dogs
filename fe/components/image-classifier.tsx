"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Upload, Loader2, Sparkles, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import Image from "next/image"

interface ExplainResult {
  method?: string
  imageUrl?: string
}

export default function ImageClassifier() {
  const [image, setImage] = useState<string | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [results, setResults] = useState<ExplainResult[]>([])
  const [label, setLabel] = useState<string | null>(null)
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
      setLabel(null)

      // Simulate Explain
      setIsClassifying(true)
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Mock result - Replace with actual API call
      const mockResults: ExplainResult[] = [
        { method: "Grad-CAM", imageUrl },
        { method: "SHAP", imageUrl },
        { method: "LIME", imageUrl },
      ]
      
      setLabel("Golden Retriever")
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
    setLabel(null)
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
              <div className="space-y-4">
                <h3 className="text-xl font-bold text-center">Picture</h3>
                <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-secondary">
                  <Image src={image || "/placeholder.svg"} alt="Original image" fill className="object-contain" />
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
        <>
          {isClassifying ? (
            <Card className="border-2 border-border bg-card p-6">
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <p className="mt-4 text-muted-foreground">Analyzing image...</p>
              </div>
            </Card>
          ) : results.length > 0 ? (
            <div className="space-y-6">
              <Card 
                className="overflow-hidden border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors"
              >
                <div className="p-4">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="rounded-full bg-primary/10 p-2">
                      <Sparkles className="h-5 w-5 text-primary"/>
                    </div>
                    <h2 className="text-2xl font-semibold">Classification Result</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-primary">{label}</span>
                  </div>
                </div>
              </Card>

              {results.map((result, index) => (
                <Card 
                  key={index}
                  className="overflow-hidden border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors"
                >
                  <div className="p-4">
                    <div className="mb-4 flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                        <span className="text-lg font-bold text-primary">{String.fromCharCode(65 + index)}</span>
                      </div>
                      <h3 className="text-xl font-semibold">{result.method}</h3>
                    </div>
                    
                    <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-secondary">
                      <Image 
                        src={result.imageUrl || image || "/placeholder.svg"} 
                        alt={`${result.method} result`} 
                        fill 
                        className="object-contain" 
                      />
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}