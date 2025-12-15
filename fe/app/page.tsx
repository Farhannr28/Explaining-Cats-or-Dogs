import ImageClassifier from "@/components/image-classifier"

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 md:py-16">
        <div className="mx-auto max-w-4xl">
          <div className="mb-12 text-center">
            <h1 className="mb-4 text-4xl font-bold tracking-tight text-balance md:text-5xl lg:text-6xl">
              AI Image <span className="text-primary">Classifier</span>
            </h1>
            <p className="text-lg text-muted-foreground text-pretty md:text-xl">
              Upload an image and let our AI model identify what&apos;s in it with stunning accuracy
            </p>
          </div>
          <ImageClassifier />
        </div>
      </div>
    </main>
  )
}
