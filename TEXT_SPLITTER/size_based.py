from langchain.text_splitter import CharacterTextSplitter

text= """ The current situation of Indian GDP and economics reflects a phase of steady recovery and growth, following global disruptions caused by the COVID-19 pandemic. As of 2025:

GDP Growth: India's GDP is projected to grow at a moderate to strong pace, often estimated around 6-7% annually, making it one of the fastest-growing major economies globally. This growth is supported by robust domestic demand, improving industrial output, and expanding services sector.

Inflation and Monetary Policy: Inflationary pressures have moderated due to improved supply chains and government interventions. The Reserve Bank of India has calibrated interest rates to balance growth with price stability.

Sector Performance: Key sectors such as manufacturing, agriculture, and information technology continue to contribute significantly to GDP. The digital economy and startup ecosystem are also rapidly expanding, driving innovation and job creation.

Foreign Investment: India continues to attract strong foreign direct investment (FDI) inflows, supported by reforms and initiatives like "Make in India" and infrastructure development.

Challenges: Despite positive trends, challenges remain including unemployment concerns, inflation risks, fiscal deficits, and structural reforms in labor and land sectors. "
"""
splitter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result=splitter.split_text(text)
print(result)